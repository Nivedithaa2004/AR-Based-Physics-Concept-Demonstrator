#!/usr/bin/env python3
"""
AR-Based Physics Concept Demonstrator
=====================================
Subject : Augmented Reality / Virtual Reality (BCG702) - Individual CCA
Tools   : Python, OpenCV (ArUco), NumPy, laptop webcam

Pipeline (one pass per video frame)
    1. Webcam capture
    2. ArUco marker detection              (cv2.aruco.ArucoDetector)
    3. Pose estimation                     (cv2.solvePnPGeneric, IPPE_SQUARE)
    4. Tracking / temporal filtering       (pose smoothing, ambiguity resolution,
                                            short-loss hold, centre trail)
    5. Registration                        (cv2.projectPoints: marker coordinates
                                            -> image pixels)
    6. AR visualisation                    (physics model rendered in marker space)

Marker IDs (DICT_4X4_50)
    ID 0 -> Projectile Motion        ID 2 -> Inclined Plane & Forces
    ID 1 -> Simple Pendulum          ID 3 -> Transverse Wave Motion

Keys:  W/S, A/D change parameters | SPACE pause | R reset | P wall/table mode
       G screenshot | V start/stop video recording | Q or ESC quit
"""
import argparse
import math
import os
import time
from collections import deque

import cv2
import numpy as np

G = 9.81  # m/s^2

# ----------------------------------------------------------------------------
# Colours (BGR)
# ----------------------------------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (60, 60, 255)
GREEN = (80, 220, 80)
BLUE = (255, 150, 60)
YELLOW = (0, 230, 255)
ORANGE = (0, 150, 255)
CYAN = (255, 230, 0)
MAGENTA = (255, 80, 220)
GREY = (190, 190, 190)


# ----------------------------------------------------------------------------
# Scene: converts marker-space geometry to pixels and draws it (REGISTRATION)
# ----------------------------------------------------------------------------
class Scene:
    """
    World frame = marker frame, unit = ONE MARKER SIDE, origin at marker centre.
    Physics content is described in a 2-D (u, v) sheet plus a small depth d:
      mode 'wall'  : sheet is the marker plane   -> (x, y, z) = (u, v, d)
                     (marker held upright / stuck on a wall; v is "up")
      mode 'table' : sheet is vertical, standing on the marker -> (u, d, v)
                     (marker lying flat on a desk; marker z axis is "up")
    """

    def __init__(self, img, K, dist, rvec, tvec, mode):
        self.img, self.K, self.dist = img, K, dist
        self.rvec, self.tvec, self.mode = rvec, tvec, mode

    def pt(self, u, v, d=0.0):
        return (u, v, d) if self.mode == "wall" else (u, d, v)

    def proj(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        out, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return out.reshape(-1, 2)

    def p2(self, p3):
        return tuple(int(round(x)) for x in self.proj([p3])[0])

    # --- primitives -------------------------------------------------------
    def line(self, a, b, color, th=2):
        pa, pb = self.p2(a), self.p2(b)
        cv2.line(self.img, pa, pb, color, th, cv2.LINE_AA)

    def polyline(self, pts, color, th=2, closed=False):
        p = np.round(self.proj(pts)).astype(np.int32)
        cv2.polylines(self.img, [p], closed, color, th, cv2.LINE_AA)

    def fill(self, pts, color, alpha=0.35):
        p = np.round(self.proj(pts)).astype(np.int32)
        ov = self.img.copy()
        cv2.fillPoly(ov, [p], color)
        cv2.addWeighted(ov, alpha, self.img, 1 - alpha, 0, self.img)
        cv2.polylines(self.img, [p], True, color, 2, cv2.LINE_AA)

    def dashed(self, a, b, color, th=1, seg=8):
        pa, pb = np.array(self.p2(a), float), np.array(self.p2(b), float)
        n = int(np.linalg.norm(pb - pa) // seg)
        for i in range(0, n, 2):
            s = pa + (pb - pa) * i / max(n, 1)
            e = pa + (pb - pa) * min(i + 1, n) / max(n, 1)
            cv2.line(self.img, tuple(s.astype(int)), tuple(e.astype(int)), color, th, cv2.LINE_AA)

    def arrow(self, a, b, color, th=3):
        pa, pb = self.p2(a), self.p2(b)
        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) < 3:
            return
        cv2.arrowedLine(self.img, pa, pb, color, th, cv2.LINE_AA, tipLength=0.22)

    def disc(self, c, r_units, color, fill=True, th=2):
        c1 = self.proj([c])[0]
        c2 = self.proj([(c[0] + r_units, c[1], c[2])])[0]
        r = max(2, int(np.linalg.norm(c2 - c1)))
        cv2.circle(self.img, (int(c1[0]), int(c1[1])), r, color, -1 if fill else th, cv2.LINE_AA)
        if fill:
            cv2.circle(self.img, (int(c1[0]), int(c1[1])), r, BLACK, 1, cv2.LINE_AA)

    def label(self, p3, text, color=WHITE, scale=0.55, dx=6, dy=-6):
        x, y = self.p2(p3)
        put_text(self.img, text, (x + dx, y + dy), color, scale)


def put_text(img, text, org, color=WHITE, scale=0.55, th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Physics concepts (AR VISUALISATION)
# ----------------------------------------------------------------------------
class Concept:
    name = ""
    controls = ""

    def __init__(self):
        self.t = 0.0
        self.paused = False
        self.init_params()
        self.reset()

    def init_params(self): ...
    def reset(self): self.t = 0.0
    def adjust(self, a, b): ...
    def advance(self, dt):
        if not self.paused:
            self.update(dt)
    def update(self, dt): self.t += dt
    def draw(self, sc): return []


class Projectile(Concept):
    name = "Projectile Motion"
    controls = "W/S: launch angle   A/D: launch speed"
    K = 0.45  # marker units per metre

    def init_params(self):
        self.angle, self.v0 = 45.0, 6.0

    def adjust(self, a, b):
        self.angle = float(np.clip(self.angle + 5 * a, 10, 85))
        self.v0 = float(np.clip(self.v0 + 0.5 * b, 2, 8))
        self.reset()

    def draw(self, sc):
        th = math.radians(self.angle)
        vx, vy = self.v0 * math.cos(th), self.v0 * math.sin(th)
        T = 2 * vy / G
        R = self.v0 ** 2 * math.sin(2 * th) / G
        H = vy ** 2 / (2 * G)
        if self.t > T + 0.7:
            self.t = 0.0
        tt = min(self.t, T)
        ox, oy, k = -0.5, -0.5, self.K
        pos = lambda t: sc.pt(ox + k * vx * t, oy + k * (vy * t - 0.5 * G * t * t))

        # ground line with 1 m ticks
        sc.line(sc.pt(ox - 0.3, oy), sc.pt(ox + k * 9, oy), GREY, 2)
        for m in range(0, 9):
            sc.line(sc.pt(ox + k * m, oy), sc.pt(ox + k * m, oy - 0.07), GREY, 1)
            sc.label(sc.pt(ox + k * m, oy - 0.07), f"{m}", GREY, 0.4, -4, 14)
        # predicted path (dotted)
        for i in range(0, 41, 2):
            sc.disc(pos(T * i / 40), 0.012, YELLOW)
        # launch-angle arc
        arc = [sc.pt(ox + 0.35 * math.cos(a), oy + 0.35 * math.sin(a)) for a in np.linspace(0, th, 12)]
        sc.polyline(arc, CYAN, 2)
        sc.label(sc.pt(ox + 0.38, oy + 0.1), f"{self.angle:.0f} deg", CYAN, 0.5)
        # apex / range markers
        apex = sc.pt(ox + k * vx * T / 2, oy + k * H)
        sc.dashed(apex, sc.pt(ox + k * vx * T / 2, oy), ORANGE)
        sc.label(apex, f"H={H:.2f} m", ORANGE, 0.5, 4, -8)
        sc.label(sc.pt(ox + k * R, oy), f"R={R:.2f} m", GREEN, 0.5, -30, -10)
        # ball + velocity vector and components
        ball = pos(tt)
        sc.disc(ball, 0.05, RED)
        vyt = vy - G * tt
        s = 0.07
        b = (ball[0], ball[1], ball[2]) if sc.mode == "wall" else ball
        tip_x = sc.pt(ox + k * vx * tt + s * vx, oy + k * (vy * tt - 0.5 * G * tt * tt))
        tip_y = sc.pt(ox + k * vx * tt, oy + k * (vy * tt - 0.5 * G * tt * tt) + s * vyt)
        tip_v = sc.pt(ox + k * vx * tt + s * vx, oy + k * (vy * tt - 0.5 * G * tt * tt) + s * vyt)
        sc.arrow(b, tip_x, BLUE, 2)
        sc.arrow(b, tip_y, MAGENTA, 2)
        sc.arrow(b, tip_v, GREEN, 3)
        speed = math.hypot(vx, vyt)
        return [
            "PROJECTILE MOTION",
            "R = v^2 sin(2*theta) / g      H = (v sin(theta))^2 / 2g",
            f"v0={self.v0:.1f} m/s  theta={self.angle:.0f} deg  g={G}",
            f"Range R = {R:.2f} m   Max height H = {H:.2f} m   Flight T = {T:.2f} s",
            f"t={tt:.2f} s   vx={vx:.2f}  vy={vyt:+.2f}  |v|={speed:.2f} m/s",
        ]


class Pendulum(Concept):
    name = "Simple Pendulum"
    controls = "W/S: initial angle   A/D: string length"
    PIVOT_V, SCALE = 2.4, 2.0  # pivot height (marker units), units per metre

    def init_params(self):
        self.L, self.theta0 = 0.8, 40.0

    def adjust(self, a, b):
        self.theta0 = float(np.clip(self.theta0 + 5 * a, 5, 70))
        self.L = float(np.clip(self.L + 0.1 * b, 0.3, 1.5))
        self.reset()

    def reset(self):
        self.t = 0.0
        self.theta = math.radians(self.theta0)
        self.omega = 0.0
        self.trail = deque(maxlen=22)
        self.last_up, self.T_meas = None, None

    def update(self, dt):
        dt = min(dt, 0.05)
        n = int(dt / 0.002) + 1
        h = dt / n
        for _ in range(n):  # semi-implicit Euler on theta'' = -(g/L) sin(theta)
            self.omega += -(G / self.L) * math.sin(self.theta) * h
            prev = self.theta
            self.theta += self.omega * h
            self.t += h
            if prev < 0 <= self.theta:  # upward zero crossing -> one period
                if self.last_up is not None:
                    self.T_meas = self.t - self.last_up
                self.last_up = self.t

    def draw(self, sc):
        pv = self.PIVOT_V
        ln = self.L * self.SCALE
        pivot = sc.pt(0, pv)
        bob = sc.pt(ln * math.sin(self.theta), pv - ln * math.cos(self.theta))
        self.trail.append(bob)
        # support
        sc.line(sc.pt(-0.7, pv), sc.pt(0.7, pv), GREY, 3)
        for i in range(-3, 4):
            sc.line(sc.pt(i * 0.2, pv), sc.pt(i * 0.2 + 0.1, pv + 0.1), GREY, 1)
        # reference vertical + angle arc
        sc.dashed(pivot, sc.pt(0, pv - ln), GREY)
        r = 0.45
        a0, a1 = -math.pi / 2, -math.pi / 2 + self.theta
        arc = [sc.pt(r * math.cos(a), pv + r * math.sin(a)) for a in np.linspace(a0, a1, 10)]
        sc.polyline(arc, CYAN, 2)
        # fading trail
        for i, p in enumerate(self.trail):
            sc.disc(p, 0.012 + 0.002 * i / 4, YELLOW)
        sc.line(pivot, bob, WHITE, 2)
        sc.disc(bob, 0.09, RED)
        # tangential velocity vector
        v = self.L * self.omega
        tvec = (math.cos(self.theta), math.sin(self.theta))
        sc.arrow(bob, sc.pt(ln * math.sin(self.theta) + 0.12 * v * tvec[0],
                            pv - ln * math.cos(self.theta) + 0.12 * v * tvec[1]), GREEN, 3)
        sc.label(sc.pt(0.05, pv - ln * 0.5), f"L={self.L:.1f} m", WHITE, 0.5)
        sc.label(sc.pt(0.12, pv - 0.55), f"{math.degrees(self.theta):+.0f} deg", CYAN, 0.5)
        T0 = 2 * math.pi * math.sqrt(self.L / G)
        pe = G * self.L * (1 - math.cos(self.theta))
        ke = 0.5 * (self.L * self.omega) ** 2
        meas = f"{self.T_meas:.2f} s" if self.T_meas else "..."
        return [
            "SIMPLE PENDULUM",
            "T = 2*pi*sqrt(L/g)    E = KE + PE = const (m = 1 kg)",
            f"L={self.L:.1f} m  theta0={self.theta0:.0f} deg  Predicted T={T0:.2f} s",
            f"Measured period (from simulation) = {meas}",
            f"KE={ke:.2f} J  PE={pe:.2f} J  E={ke + pe:.2f} J  v={self.L * self.omega:+.2f} m/s",
        ]


class Incline(Concept):
    name = "Inclined Plane & Forces"
    controls = "W/S: incline angle   A/D: friction coefficient"
    LR, SCALE_F = 2.8, 0.09  # slope length (units), arrow units per newton

    def init_params(self):
        self.angle, self.mu = 30.0, 0.20

    def adjust(self, a, b):
        self.angle = float(np.clip(self.angle + 5 * a, 5, 60))
        self.mu = float(np.clip(self.mu + 0.05 * b, 0.0, 1.0))
        self.reset()

    def draw(self, sc):
        th = math.radians(self.angle)
        c, s = math.cos(th), math.sin(th)
        A = (-1.4, -0.5)
        Bp = (A[0] + self.LR * c, A[1])
        Cp = (A[0] + self.LR * c, A[1] + self.LR * s)
        sc.fill([sc.pt(*A), sc.pt(*Bp), sc.pt(*Cp)], (120, 160, 120), 0.35)
        t_hat, n_hat = np.array([c, s]), np.array([-s, c])
        a = G * (s - self.mu * c)
        sliding = a > 1e-6
        s0 = self.LR - 0.3
        s_end = 0.3
        if sliding:
            t_end = math.sqrt(2 * (s0 - s_end) / a)
            if self.t > t_end + 0.8:
                self.t = 0.0
            dist = min(0.5 * a * self.t ** 2, s0 - s_end)
        else:
            dist = 0.0
        pos = np.array(A) + (s0 - dist) * t_hat
        ctr = pos + n_hat * 0.18
        # block (square aligned with slope)
        h = 0.18
        blk = [ctr + t_hat * dx + n_hat * dy for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        sc.fill([sc.pt(*p) for p in blk], ORANGE, 0.6)
        W = G  # m = 1 kg
        N = W * c
        f = min(self.mu * N, W * s) if not sliding else self.mu * N
        F = self.SCALE_F
        o = sc.pt(*ctr)
        sc.arrow(o, sc.pt(ctr[0], ctr[1] - F * W), RED, 3)
        sc.label(sc.pt(ctr[0], ctr[1] - F * W), "mg", RED, 0.55, 4, 14)
        e = ctr + n_hat * F * N
        sc.arrow(o, sc.pt(*e), GREEN, 3)
        sc.label(sc.pt(*e), "N", GREEN, 0.55, 4, -4)
        e = ctr - t_hat * F * W * s
        sc.arrow(o, sc.pt(*e), MAGENTA, 3)
        sc.label(sc.pt(*e), "mg sin", MAGENTA, 0.45, -48, 14)
        if f > 0.05:
            e = ctr + t_hat * F * f
            sc.arrow(o, sc.pt(*e), YELLOW, 3)
            sc.label(sc.pt(*e), "f", YELLOW, 0.55, 4, -4)
        sc.dashed(o, sc.pt(*(ctr - n_hat * F * W * c)), RED)
        arc = [sc.pt(A[0] + 0.55 * math.cos(x), A[1] + 0.55 * math.sin(x)) for x in np.linspace(0, th, 10)]
        sc.polyline(arc, CYAN, 2)
        sc.label(sc.pt(A[0] + 0.58, A[1] + 0.12), f"{self.angle:.0f} deg", CYAN, 0.5)
        state = (f"SLIDING down, a = {a:.2f} m/s^2" if sliding
                 else "STATIC: friction balances mg sin(theta)")
        return [
            "INCLINED PLANE & FORCES",
            "N = mg cos(theta)   f = mu N   a = g (sin(theta) - mu cos(theta))",
            f"theta={self.angle:.0f} deg  mu={self.mu:.2f}  (m = 1 kg)",
            f"mg={W:.2f} N  N={N:.2f} N  mg sin={W * s:.2f} N  f={f:.2f} N",
            state + f"   (tan theta={math.tan(th):.2f} vs mu={self.mu:.2f})",
        ]


class Wave(Concept):
    name = "Transverse Wave Motion"
    controls = "W/S: wavelength   A/D: frequency"
    U0, V0, AMP = 1.7, 0.95, 0.28

    def init_params(self):
        self.lam, self.freq = 1.0, 1.0

    def adjust(self, a, b):
        self.lam = float(np.clip(self.lam + 0.25 * a, 0.5, 2.0))
        self.freq = float(np.clip(self.freq + 0.25 * b, 0.5, 2.0))

    def draw(self, sc):
        k, w = 2 * math.pi / self.lam, 2 * math.pi * self.freq
        U0, V0, A = self.U0, self.V0, self.AMP
        y = lambda x: V0 + A * math.sin(k * x - w * self.t)
        xs = np.linspace(-U0, U0, 90)
        sc.dashed(sc.pt(-U0, V0), sc.pt(U0, V0), GREY)
        sc.polyline([sc.pt(x, y(x)) for x in xs], CYAN, 3)
        # medium particles oscillate vertically
        for x in np.arange(-U0, U0 + 0.01, 0.25):
            p = sc.pt(x, y(x))
            sc.disc(p, 0.03, RED)
            vy = -A * w * math.cos(k * x - w * self.t)  # particle velocity
            if abs(x % 0.5) < 1e-6:
                sc.arrow(p, sc.pt(x, y(x) + 0.06 * vy), GREEN, 2)
        # wavelength bracket between two successive crests
        n = math.ceil((k * (-U0) - math.pi / 2 - (-w * self.t) * -1) / (2 * math.pi))
        x0 = (math.pi / 2 + w * self.t + 2 * math.pi * n) / k
        while x0 < -U0:
            x0 += self.lam
        if x0 + self.lam > U0:
            x0 -= self.lam
        yb = V0 + A + 0.22
        sc.arrow(sc.pt(x0, yb), sc.pt(x0 + self.lam, yb), YELLOW, 2)
        sc.arrow(sc.pt(x0 + self.lam, yb), sc.pt(x0, yb), YELLOW, 2)
        sc.label(sc.pt(x0 + self.lam / 2, yb), f"lambda={self.lam:.2f} m", YELLOW, 0.5, -45, -8)
        sc.arrow(sc.pt(U0 - 0.5, V0 - 0.55), sc.pt(U0 - 0.05, V0 - 0.55), MAGENTA, 3)
        sc.label(sc.pt(U0 - 0.9, V0 - 0.55), "wave", MAGENTA, 0.5, 0, -8)
        return [
            "TRANSVERSE WAVE",
            "y(x,t) = A sin(kx - wt)     v = f * lambda",
            f"lambda={self.lam:.2f} m   f={self.freq:.2f} Hz   A=0.30 (display)",
            f"Wave speed v = {self.freq * self.lam:.2f} m/s   Period T = {1 / self.freq:.2f} s",
            f"k = 2pi/lambda = {k:.2f} rad/m   w = 2pi f = {w:.2f} rad/s",
        ]


CONCEPTS = {0: Projectile, 1: Pendulum, 2: Incline, 3: Wave}


# ----------------------------------------------------------------------------
# Application: capture -> detect -> pose -> track -> register -> visualise
# ----------------------------------------------------------------------------
class ARPhysicsApp:
    MAX_MISSED = 6      # frames a marker may vanish before the track is dropped
    SMOOTH = 0.55       # weight of the newest pose (1 = no smoothing)

    def __init__(self, frame_size, calib=None, mode="wall", marker_cm=5.0):
        w, h = frame_size
        if calib and os.path.exists(calib):
            d = np.load(calib)
            self.K, self.dist = d["K"], d["dist"]
            self.calibrated = True
        else:  # approximate pinhole model: f ~ image width, centre principal point
            self.K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
            self.dist = np.zeros(5)
            self.calibrated = False
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self.obj = np.array([[-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0], [-.5, -.5, 0]], np.float32)
        self.tracks, self.concepts = {}, {}
        self.mode, self.marker_cm = mode, marker_cm
        self.active = None
        self.fps, self._t_prev = 0.0, time.time()

    # -- step 3: pose estimation -------------------------------------------
    def estimate_pose(self, corners, mid):
        n, rvecs, tvecs, err = cv2.solvePnPGeneric(
            self.obj, corners.reshape(4, 2).astype(np.float32), self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if n == 0:
            return None
        prev = self.tracks.get(mid)
        if prev is not None:  # resolve planar-pose ambiguity using temporal continuity
            i = int(np.argmin([np.linalg.norm(r - prev["rvec"]) for r in rvecs]))
        else:
            i = int(np.argmin(err))
        return rvecs[i], tvecs[i]

    # -- step 4: tracking ---------------------------------------------------
    def track(self, mid, rvec, tvec, centre):
        tr = self.tracks.get(mid)
        if tr is None:
            self.tracks[mid] = dict(rvec=rvec, tvec=tvec, missed=0, trail=deque(maxlen=40))
            tr = self.tracks[mid]
        else:
            a = self.SMOOTH
            if np.linalg.norm(rvec - tr["rvec"]) < 0.8:
                tr["rvec"] = a * rvec + (1 - a) * tr["rvec"]
            else:
                tr["rvec"] = rvec
            tr["tvec"] = a * tvec + (1 - a) * tr["tvec"]
            tr["missed"] = 0
        tr["trail"].append(centre)

    @staticmethod
    def euler_deg(rvec):
        R, _ = cv2.Rodrigues(rvec)
        sy = math.hypot(R[0, 0], R[1, 0])
        pitch = math.atan2(-R[2, 1], -R[2, 2])  # 0 deg = marker facing camera
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
        return [math.degrees(x) for x in (pitch, yaw, roll)]

    def process_frame(self, frame, dt=None):
        now = time.time()
        if dt is None:
            dt = now - self._t_prev
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-3) if self.fps else 1 / max(dt, 1e-3)
        self._t_prev = now
        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # step 2: detection
        corners, ids, _ = self.detector.detectMarkers(gray)
        seen = set()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                mid = int(i)
                if mid not in CONCEPTS:
                    put_text(out, f"ID {mid}: no concept assigned", tuple(c[0][0].astype(int)), RED)
                    continue
                pose = self.estimate_pose(c, mid)
                if pose is None:
                    continue
                self.track(mid, pose[0], pose[1], tuple(c[0].mean(axis=0).astype(int)))
                seen.add(mid)
        for mid in list(self.tracks):
            if mid not in seen:
                self.tracks[mid]["missed"] += 1
                if self.tracks[mid]["missed"] > self.MAX_MISSED:
                    del self.tracks[mid]

        # steps 5 & 6: registration + visualisation
        hud, pose_txt = None, None
        for mid in sorted(self.tracks):
            tr = self.tracks[mid]
            con = self.concepts.setdefault(mid, CONCEPTS[mid]())
            if mid in seen:
                con.advance(dt)
            sc = Scene(out, self.K, self.dist, tr["rvec"], tr["tvec"], self.mode)
            cv2.drawFrameAxes(out, self.K, self.dist, tr["rvec"], tr["tvec"], 0.3, 2)
            for a, b in zip(list(tr["trail"])[:-1], list(tr["trail"])[1:]):
                cv2.line(out, a, b, (200, 200, 0), 1, cv2.LINE_AA)
            lines = con.draw(sc)
            if hud is None:
                hud, self.active = (con, lines), mid
                pitch, yaw, roll = self.euler_deg(tr["rvec"])
                dist_cm = float(np.linalg.norm(tr["tvec"])) * self.marker_cm
                pose_txt = [f"Marker ID {mid}  [{'TRACKING' if mid in seen else 'HOLDING'}]",
                            f"Distance: {dist_cm:.0f} cm",
                            f"Pitch {pitch:+.0f}  Yaw {yaw:+.0f}  Roll {roll:+.0f} deg"]
        if not self.tracks:
            self.active = None
        self.draw_hud(out, hud, pose_txt)
        return out

    def draw_hud(self, img, hud, pose_txt):
        H, W = img.shape[:2]
        if hud:
            con, lines = hud
            panel(img, 8, 8, 640, 14 + 25 * len(lines))
            for i, ln in enumerate(lines):
                put_text(img, ln, (18, 32 + 25 * i), YELLOW if i == 0 else WHITE, 0.62 if i == 0 else 0.52)
            panel(img, W - 330, 8, 322, 90)
            for i, ln in enumerate(pose_txt):
                put_text(img, ln, (W - 320, 32 + 24 * i), GREEN if i == 0 else WHITE, 0.52)
            ctrl = con.controls + "   SPACE pause  R reset  P wall/table"
        else:
            panel(img, 8, 8, 640, 64)
            put_text(img, "Show an ArUco marker (ID 0-3, DICT_4X4_50) to the camera", (18, 32), YELLOW, 0.62)
            put_text(img, "0 Projectile | 1 Pendulum | 2 Inclined plane | 3 Wave", (18, 58), WHITE, 0.52)
            ctrl = ""
        panel(img, 8, H - 34, W - 16, 26)
        put_text(img, f"{ctrl}   |   G shot  V record  Q quit   |   {self.fps:.0f} FPS  "
                      f"mode={self.mode}  {'calibrated' if self.calibrated else 'approx. K'}",
                 (16, H - 15), GREY, 0.45)

    def handle_key(self, k):
        con = self.concepts.get(self.active)
        if k == ord('p'):
            self.mode = "table" if self.mode == "wall" else "wall"
        if con is None:
            return
        if k in (ord('w'), ord('s')):
            con.adjust(1 if k == ord('w') else -1, 0)
        elif k in (ord('a'), ord('d')):
            con.adjust(0, 1 if k == ord('d') else -1)
        elif k == ord(' '):
            con.paused = not con.paused
        elif k == ord('r'):
            con.reset()


def panel(img, x, y, w, h, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def open_camera(idx, w, h):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def main():
    ap = argparse.ArgumentParser(description="AR-Based Physics Concept Demonstrator")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--calib", default="calibration.npz", help="optional camera calibration file")
    ap.add_argument("--mode", choices=["wall", "table"], default="wall")
    ap.add_argument("--marker-cm", type=float, default=5.0, help="printed marker side length in cm")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera/video. Try --camera 1.")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No frames received from camera.")
    h, w = frame.shape[:2]
    app = ARPhysicsApp((w, h), args.calib, args.mode, args.marker_cm)
    os.makedirs("screenshots", exist_ok=True)
    writer = None
    win = "AR Physics Concept Demonstrator"
    while ok:
        out = app.process_frame(frame)
        if writer is not None:
            writer.write(out)
            cv2.circle(out, (w - 20, h - 50), 7, (0, 0, 255), -1)
        cv2.imshow(win, out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('g'):
            fn = f"screenshots/shot_{int(time.time())}.png"
            cv2.imwrite(fn, out)
            print("saved", fn)
        elif k == ord('v'):
            if writer is None:
                writer = cv2.VideoWriter(f"demo_{int(time.time())}.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                print("recording...")
            else:
                writer.release()
                writer = None
                print("recording saved")
        elif k != 255:
            app.handle_key(k)
        ok, frame = cap.read()
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
