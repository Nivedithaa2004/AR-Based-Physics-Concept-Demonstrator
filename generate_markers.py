"""Generate printable ArUco markers (DICT_4X4_50) for the four physics concepts."""
import cv2, numpy as np, os
os.makedirs("markers", exist_ok=True)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
names = {0: "Projectile Motion", 1: "Simple Pendulum", 2: "Inclined Plane", 3: "Wave Motion"}
for i, n in names.items():
    m = cv2.aruco.generateImageMarker(d, i, 400)
    img = np.full((520, 480), 255, np.uint8)          # white quiet zone is REQUIRED
    img[40:440, 40:440] = m
    cv2.putText(img, f"ID {i}: {n}", (40, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    cv2.imwrite(f"markers/marker_{i}.png", img)
print("Markers written to ./markers  (print ~5 cm wide, keep the white border)")
