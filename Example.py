from ObjectDetector import ObjectDetector
import cv2
import numpy as np
from datetime import datetime

detector = ObjectDetector(
    model_path="model_final.pth",
    num_classes=1,
    confidence_threshold=0.7,
    device="cpu"
)

T = [[-0.99680019, -0.03578532, 0.07147588, 0.33126804],
     [0.03749214, 0.5804158, 0.81345672, -0.1964696],
     [-0.07059553, 0.8135336, -0.57721691, 0.59237415],
     [0., 0., 0., 1.]]

T1 = np.array([[1, 0, 0, 0.406],
               [0, 1, 0, -0.569],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

T_inv = np.linalg.inv(T)

counter = 0

while True:
    print(f"Frame: {counter}")
    detector.get_frame()
    masks, centers = detector.get_bitmask()
    detector.visualize()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    for i in range(len(masks)):
        x, y, z = detector.get_3d_position(masks[i], centers[i])
        print(f'Coordinates (CAMERA frame): {x * 1000 :.3f}, {y * 1000 :.3f}, {z * 1000 :.3f}')

        point_camera = np.array([x, y, z, 1])
        point_target = np.dot(T_inv, point_camera)
        point_base = np.dot(T1, point_target)

        print(
            f'Coordinates (TARGET frame): {point_target[0] * 1000:.3f}, {point_target[1] * 1000:.3f}, {point_target[2] * 1000:.3f}')
        print(
            f'Coordinates (BASE frame): {point_base[0] * 1000 :.3f}, {point_base[1] * 1000 :.3f}, {point_base[2] * 1000:.3f}')


    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.release()
cv2.destroyAllWindows()


