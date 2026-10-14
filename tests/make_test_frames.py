"""
Offline test harness: renders synthetic 'webcam' frames that contain a printed
ArUco marker in different poses (visible / moved / rotated) and runs the full
AR pipeline on them. Used to verify detection, pose estimation and registration
without a camera, and to produce Test-case screenshots.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from ar_physics_demo import ARPhysicsApp

W, H = 1280, 720

def background():
    y, x = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = 150 + 40 * y / H; bg[..., 1] = 140 + 40 * y / H; bg[..., 2] = 125 + 30 * y / H
    noise = np.random.RandomState(1).randint(-6, 6, (H, W, 1))
    return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

def place(marker_id, centre, size, yaw=0.0, tilt=0.0, bg=None):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, marker_id, 400)
    sheet = np.full((520, 520), 255, np.uint8); sheet[60:460, 60:460] = m
    sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    s = size * 520 / 400
    pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * s / 2
    a = np.radians(yaw); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ R.T
    t = np.radians(tilt)
    z = 1 + 0.0012 * pts[:, 0] * np.sin(t) * 3
    pts = np.stack([pts[:, 0] * np.cos(t) / z, pts[:, 1] / z], 1) + centre
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [520, 0], [520, 520], [0, 520]]), pts.astype(np.float32))
    warped = cv2.warpPerspective(sheet, Hm, (W, H))
    mask = cv2.warpPerspective(np.full((520, 520), 255, np.uint8), Hm, (W, H))
    out = bg.copy(); out[mask > 0] = warped[mask > 0]
    return cv2.GaussianBlur(out, (3, 3), 0)

def run(marker_id, centre, size, yaw, tilt, name, warm=14, mode="wall"):
    app = ARPhysicsApp((W, H), None, mode)
    frame = place(marker_id, centre, size, yaw, tilt, background())
    for _ in range(warm):
        res = app.process_frame(frame, dt=0.05)
    cv2.imwrite(f"screenshots/{name}.png", res)
    return app

if __name__ == "__main__":
    os.makedirs("screenshots", exist_ok=True)
    for mid, nm in enumerate(["projectile", "pendulum", "incline", "wave"]):
        run(mid, (560, 470), 200, 0, 0, f"{nm}_visible")
        run(mid, (800, 400), 170, 0, 0, f"{nm}_moved")
        run(mid, (640, 440), 190, 25, 30, f"{nm}_rotated")
    f = place(0, (640, 400), 220, 15, 20, background())
    app = ARPhysicsApp((W, H), None)
    c, ids, _ = app.detector.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cv2.aruco.drawDetectedMarkers(f, c, ids)
    cv2.imwrite("screenshots/detection_only.png", f)
    print("detected ids:", ids.flatten() if ids is not None else None)
