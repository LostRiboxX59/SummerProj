from RealsenseCamera import RealSenseCamera
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from typing import List, Tuple, Optional, Union


class ObjectDetector:
    def __init__(self, model_path: str, num_classes: int = 1,
                 confidence_threshold: float = 0.8, device: str = "cpu"):

        # Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)

        # RealSense
        self.camera = RealSenseCamera()
        if not self.camera.start():
            raise RuntimeError("Failed to initialize RealSense camera")

        # DATA
        self._current_frame = None
        self._current_depth_frame = None
        self._current_masks = []
        self._current_centers = []

    def get_frame(self) -> bool:
        result = self.camera.get_frames()
        if result:
            self._current_frame, depth_image, self._current_depth_frame = result
            return True
        return False

    def process_image(self, image: Union[np.ndarray, str]) -> bool:
        if isinstance(image, str):
            self._current_frame = cv2.imread(image)
            if self._current_frame is None:
                print(f"Failed to load image from {image}")
                return False
        else:
            self._current_frame = image.copy()

        self._current_depth_frame = None
        return True

    def get_bitmask(self, target_class_id: int = 0) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        if self._current_frame is None:
            if not self.get_frame():
                return [], []

        outputs = self.predictor(self._current_frame)
        instances = outputs["instances"].to("cpu")

        self._current_masks = []
        self._current_centers = []

        for i in range(len(instances)):
            if instances.pred_classes[i] != target_class_id:
                continue

            mask = instances.pred_masks[i].numpy().astype(np.uint8)
            self._current_masks.append(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    self._current_centers.append((cx, cy))

        return self._current_masks, self._current_centers

    def visualize(self, show_centers: bool = True, display: bool = True,
                  window_name: str = "Object Detection") -> Optional[np.ndarray]:

        if self._current_frame is None:
            if not self.get_frame():
                return None

        vis_frame = self._current_frame.copy()

        green_color = np.array([0, 255, 0], dtype=np.uint8)  # Создаем numpy array

        for mask in self._current_masks:

            vis_frame[mask > 0] = (vis_frame[mask > 0] * 0.7 + green_color * 0.3).astype(np.uint8)

        if show_centers:
            for (cx, cy) in self._current_centers:
                cv2.circle(vis_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(vis_frame, (cx, cy), 10, (255, 255, 255), 2)

        if display:
            cv2.imshow(window_name, vis_frame)
            cv2.waitKey(1)
            return vis_frame
        return vis_frame

    def get_3d_position(self, mask: np.ndarray, center_pixel: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:

        if self._current_depth_frame is None:
            print("Depth data not available")
            return None

        avg_depth = self.camera.get_average_depth_in_mask(self._current_depth_frame, mask)
        if avg_depth is None:
            return None

        cx, cy = center_pixel
        x = (cx - self.camera.depth_intrinsics.ppx) * avg_depth / self.camera.depth_intrinsics.fx
        y = (cy - self.camera.depth_intrinsics.ppy) * avg_depth / self.camera.depth_intrinsics.fy
        z = avg_depth

        return x, y, z

    def _pca_orientation(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:

        y, x = np.where(mask > 0)
        points = np.column_stack((x, y))

        mean = np.mean(points, axis=0)
        points_centered = points - mean

        cov = np.cov(points_centered.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]

        main_axis = eigenvectors[:, 0]
        secondary_axis = eigenvectors[:, 1]

        angle = np.arctan2(main_axis[1], main_axis[0])

        return main_axis, secondary_axis, angle

    def get_orientation(self, mask_index: int = 0) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:

        if not self._current_masks or mask_index >= len(self._current_masks):
            return None

        mask = self._current_masks[mask_index]
        main_axis, secondary_axis, angle = self._pca_orientation(mask)
        return angle, main_axis, secondary_axis

    def visualize_with_orientation(self, display: bool = True,
                                   window_name: str = "Orientation Detection") -> Optional[np.ndarray]:

        if self._current_frame is None:
            if not self.get_frame():
                return None

        vis_frame = self._current_frame.copy()
        green_color = np.array([0, 255, 0], dtype=np.uint8)

        for i, mask in enumerate(self._current_masks):
            vis_frame[mask > 0] = (vis_frame[mask > 0] * 0.7 + green_color * 0.3).astype(np.uint8)

            orientation = self.get_orientation(i)
            if orientation is None:
                continue

            angle, main_axis, secondary_axis = orientation
            center = self._current_centers[i]

            axis_length = 50
            main_end = (int(center[0] + main_axis[0] * axis_length),
                        int(center[1] + main_axis[1] * axis_length))
            secondary_end = (int(center[0] + secondary_axis[0] * axis_length),
                             int(center[1] + secondary_axis[1] * axis_length))

            cv2.arrowedLine(vis_frame, center, main_end, (0, 0, 255), 2)
            cv2.arrowedLine(vis_frame, center, secondary_end, (255, 0, 0), 2)

            cv2.putText(vis_frame, f"{np.degrees(angle):.1f} deg",
                        (center[0] + 20, center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if display:
            cv2.imshow(window_name, vis_frame)
            cv2.waitKey(1)
            return None
        return vis_frame

    def release(self):
        self.camera.stop()
        cv2.destroyAllWindows()
