# AR-Based Physics Concept Demonstrator
Python + OpenCV (ArUco) + NumPy. Runs on a laptop webcam, no internet needed.

## Setup
    pip install -r requirements.txt
    python generate_markers.py        # creates markers/marker_0..3.png -> print them (~5 cm wide, keep white border)

## Run
    python ar_physics_demo.py                 # webcam 0
    python ar_physics_demo.py --camera 1      # another camera
    python ar_physics_demo.py --mode table    # marker lying flat on desk (physics "up" = marker normal)

| Marker ID | Concept            | W / S            | A / D           |
|-----------|--------------------|------------------|-----------------|
| 0         | Projectile motion  | launch angle     | launch speed    |
| 1         | Simple pendulum    | initial angle    | string length   |
| 2         | Inclined plane     | incline angle    | friction coeff. |
| 3         | Transverse wave    | wavelength       | frequency       |

SPACE pause, R reset, P wall/table, G screenshot (screenshots/), V record video, Q/ESC quit.

## Optional calibration
    python calibrate_camera.py   # chessboard 9x6 inner corners, 2.5 cm squares -> calibration.npz (auto-loaded)

## Offline test (no camera)
    python tests/make_test_frames.py
