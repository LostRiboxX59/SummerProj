import pyrealsense2 as rs
import numpy as np
from typing import Optional, Tuple, List


class RealSenseCamera:
    def __init__(self, depth_width: int = 640, depth_height: int = 480,
                 color_width: int = 640, color_height: int = 480,
                 fps: int = 30):

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)

        #self.spatial_filter = rs.spatial_filter()
        #self.temporal_filter = rs.temporal_filter()
        #self.hole_filling_filter = rs.hole_filling_filter()

        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.original_depth_frame = None
        self.depth_sensor = None

    def start(self) -> bool:
        try:
            profile = self.pipeline.start(self.config)
            self.depth_sensor = profile.get_device().first_depth_sensor()

            # ДЕФОЛТНЫЕ НАСТРОЙКИ
            self.depth_sensor.set_option(rs.option.laser_power, 150)  # ← ТУТ МЕНЯЙ ЗНАЧЕНИЕ!
            self.depth_sensor.set_option(rs.option.emitter_enabled, 1)

            # Получение параметров камеры
            self.depth_scale = self.depth_sensor.get_depth_scale()

            depth_profile = profile.get_stream(rs.stream.depth)
            color_profile = profile.get_stream(rs.stream.color)
            self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            print("RealSense camera started with default settings")
            return True

        except Exception as e:
            print(f"Failed to start RealSense camera: {e}")
            return False

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, rs.depth_frame]]:
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            filtered_depth = depth_frame
            # filtered_depth = self.spatial_filter.process(filtered_depth)
            # filtered_depth = self.temporal_filter.process(filtered_depth)
            # filtered_depth = self.hole_filling_filter.process(filtered_depth)

            # Конвертация в numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(filtered_depth.get_data())

            return color_image, depth_image, filtered_depth

        except Exception as e:
            print(f"Error getting frames: {e}")
            return None

    def get_3d_point(self, depth_frame: rs.depth_frame, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        try:
            depth_value = depth_frame.get_distance(x, y)
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.depth_intrinsics, [x, y], depth_value
                )
                return point_3d
        except Exception as e:
            print(f"Error getting 3D point: {e}")
        return None

    def get_average_depth_in_mask(self, depth_frame: rs.depth_frame, mask: np.ndarray) -> Optional[float]:
        try:

            depth_image = np.asanyarray(depth_frame.get_data())

            y_coords, x_coords = np.where(mask > 0)

            if len(y_coords) == 0:
                return None

            depth_values = depth_image[y_coords, x_coords].astype(float)


            depth_values = depth_values * self.depth_scale

            valid_depths = depth_values[depth_values > 0]

            if len(valid_depths) == 0:
                return None

            median_depth = np.median(valid_depths)
            std_depth = np.std(valid_depths)

            filtered_depths = valid_depths[np.abs(valid_depths - median_depth) < 2 * std_depth]

            if len(filtered_depths) == 0:
                return median_depth

            return np.mean(filtered_depths)

        except Exception as e:
            print(f"Error calculating average depth: {e}")
            return None

    def stop(self):
        self.pipeline.stop()
        print("RealSense camera stopped")
