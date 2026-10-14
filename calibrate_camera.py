"""
Optional camera calibration with a printed chessboard (9x6 inner corners).
Run, then press SPACE to capture ~15 views from different angles, Q to finish.
Produces calibration.npz (K, dist) which ar_physics_demo.py loads automatically.
"""
import cv2, numpy as np
PAT, SQ = (9, 6), 2.5  # inner corners, square size in cm
objp = np.zeros((PAT[0] * PAT[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2) * SQ
cap, objs, imgs, size = cv2.VideoCapture(0), [], [], None
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); size = g.shape[::-1]
    found, c = cv2.findChessboardCorners(g, PAT)
    v = f.copy()
    if found: cv2.drawChessboardCorners(v, PAT, c, found)
    cv2.putText(v, f"captured {len(objs)}  SPACE=capture  Q=finish", (10, 30), 0, 0.7, (0, 255, 0), 2)
    cv2.imshow("calibration", v)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' ') and found:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objs.append(objp); imgs.append(c)
    elif k == ord('q'): break
if len(objs) >= 5:
    err, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    np.savez("calibration.npz", K=K, dist=dist)
    print("RMS reprojection error:", err); print("K =\n", K)
else:
    print("Need at least 5 captures.")
