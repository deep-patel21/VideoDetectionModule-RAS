import os
import cv2
import csv
import time
import logging
import argparse
import platform
import numpy as np

from queue import Queue
from threading import Thread

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from videodetectionmode_api import start_api, update_status
from videodetectionmode_api import HOST_NAME, PORT_NUMBER

# Protection against dependency errors on non-linux platforms
if os.name == "posix":
    try:
        from picamera2 import Picamera2
        PICAMERA_AVAILABLE = True
    except ImportError:
        PICAMERA_AVAILABLE = False
else:
    PICAMERA_AVAILABLE = False

LANDMARK_PREDICTION_MODEL = "models/face_landmarker.task"

LOG_OPTIONS     = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

CAMERA_INDEX              = 0
TARGET_FPS                = 24
DETECTION_SKIPPED_FRAMES  = 3
DEFAULT_EAR_THRESHOLD     = 0.17
CALIBRATION_DURATION      = 7.0     # [seconds]
OBSERVATION_SAFETY_WINDOW = 5.0     # [seconds]
OBSERVATION_LOST_TIMEOUT  = 0.8

YAW_TOLERANCE             = 22.0    # [degrees]
PITCH_TOLERANCE           = 20.0    # [degrees]

FRAME_WIDTH               = 640     # [pixels]
FRAME_HEIGHT              = 480     # [pixels]

DOWNSCALED_FRAME_WIDTH_PX  = 320    # [pixels]
DOWNSCALED_FRAME_HEIGHT_PX = 240    # [pixels]


class CameraStream:
    """
    A wrapper class to select and operate a video stream based on available dependencies.
    """

    def __init__(self, index=CAMERA_INDEX):
        self.is_picamera = False
        self.cam = None

        if PICAMERA_AVAILABLE:
            logging.info("Video Capture Method: Picamera2")
            self.setup_picamera()
        else:
            logging.info("Video Capture Method: OpenCV")
            self.setup_opencv_videocapture()

    def setup_opencv_videocapture(self):
        self.cam = cv2.VideoCapture(CAMERA_INDEX)
        logging.info("OpenCV Video Capture initialized.")

    def setup_picamera(self):
        self.cam = Picamera2()

        # Picamera configuration
        picam_config = self.cam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        self.cam.configure(picam_config)

        # Picamera controls
        self.cam.set_controls({"AfMode": 2})

        self.cam.start()
        self.is_picamera = True

        time.sleep(1.0)
        logging.info("Picamera initialized.")

    def get_camera_type(self):
        if self.is_picamera:
            return "Picamera2"
        else:
            return "OpenCV"

    def isOperational(self):
        if self.is_picamera:
            return self.cam is not None
        else:
            return self.cam is not None and self.cam.isOpened()

    def read(self):
        if self.is_picamera:
            frame = self.cam.capture_array()
            
            if frame is not None and frame.size > 0:
                return True, frame
        else:
            return self.cam.read()

    def release(self):
        if self.cam is not None:
            if self.is_picamera:
                self.cam.stop()
            else:
                self.cam.release()

    def get_dimensions(self):
        if self.is_picamera:
            return FRAME_WIDTH, FRAME_HEIGHT
        else:
            return (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class VideoLogger:
    """
    A logger class to record video processing metrics into a CSV file.
    """

    def __init__(self, active=False, buffer_size=100):
        self.active = active

        self.folder   = "csv_logs"
        self.filename = os.path.join(self.folder, f"video_log_{int(time.time())}.csv")

        self.buffer = []
        self.buffer_size = buffer_size

        if self.active:
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            with open(self.filename, "w", newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["timestamp", "eye_ar", "threshold", "head_direction", "observation_complete"])

    def record(self, eye_ar, threshold, head_direction, observation_complete):
        if self.active:
            timestamp = time.time()

            eye_ar_short = round(float(eye_ar), 3)
            self.buffer.append([timestamp, eye_ar_short, threshold, head_direction, int(observation_complete)])

            # Flush the buffer when buffer_size is filled
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()

    def flush_buffer(self):
        if self.buffer:
            with open(self.filename, "a", newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerows(self.buffer)

            # Clear the buffer after write operation
            self.buffer = []


class CalibrationManager:
    """
    A manager class to handle initial calibration process for eye aspect ratio threshold
    """

    def __init__ (self, duration):
        self.start_time          = None
        self.duration            = duration
        self.is_calibration_done = False

        # eye parameters
        self.threshold           = DEFAULT_EAR_THRESHOLD

        # head parameters
        self.yaw_base      = 0.0
        self.pitch_base    = 0.0

        self.yaw_samples   = []
        self.pitch_samples = []

        self.last_valid_yaw_delta = 0.0

        self.data = []

    def get_progress_string(self):
        if self.is_calibration_done:
            return "CALIBRATED"
        
        if self.start_time:
            elapsed_time = time.perf_counter() - self.start_time
        else:
            elapsed_time = 0.0

        progress_string = f"Calibrating EAR: {int(elapsed_time)}/{int(self.duration)}s"
        return progress_string

    def get_informed_direction(self, head_state, current_yaw, current_pitch, y_tol=YAW_TOLERANCE, p_tol=PITCH_TOLERANCE):
        if not self.is_calibration_done:
            return head_state

        if head_state != "LOST":
            yaw_delta = current_yaw - self.yaw_base
            pitch_delta = current_pitch - self.pitch_base
            self.last_valid_yaw_delta = yaw_delta

            if yaw_delta < -y_tol:
                return "LEFT"
            elif yaw_delta > y_tol:
                return "RIGHT"
            elif pitch_delta > p_tol:
                return "DOWN"
            else:
                return "FORWARD"
        else:
            if self.last_valid_yaw_delta < -18.0: 
                return "LEFT"
            return "LOST"

    def update_calibration(self, head_direction, eye_ar, yaw=None, pitch=None):
        if self.is_calibration_done:
            return True
        
        if self.start_time is None:
            self.start_time = time.perf_counter()
            logging.info(f"Calibration for eye threshold started. Duration target: {self.duration} seconds.")

        calibration_time = time.perf_counter() - self.start_time

        # update eye parameters
        if head_direction != "LOST":
            self.data.append(eye_ar)

        # update head parameters
        if yaw is not None:
            self.yaw_samples.append(yaw)
        if pitch is not None:
            self.pitch_samples.append(pitch)

        if calibration_time >= self.duration:
            if self.data:
                average_eye_open_ar = np.mean(self.data)
                
                # Set the threshold to a percentage of the average eye open ratio
                self.threshold = round(average_eye_open_ar * 0.70, 3)
                logging.info(f"Calibration completed at threshold: {self.threshold:.3f} (Average Eye Open AR: {average_eye_open_ar:.3f})")
            if self.yaw_samples:
                self.yaw_base = np.mean(self.yaw_samples)
                logging.info(f"Calibration completed for yaw base: {self.yaw_base:.3f}")
            if self.pitch_samples:
                self.pitch_base = np.mean(self.pitch_samples)
                logging.info(f"Calibration completed at pitch base: {self.pitch_base:.3f}")
                
            self.is_calibration_done = True
        
        return self.is_calibration_done


class LandmarkDetector:
    """
    A detection class to apply MediaPipe facial landmark detection and determine mesh eye and head poses
    """

    def __init__(self, landmarker_model):
        self.downscale_width  = DOWNSCALED_FRAME_WIDTH_PX
        self.downscale_height = DOWNSCALED_FRAME_HEIGHT_PX

        # Head pose stability stores
        self.direction_buffer = []
        self.buffer_size = 5

        # Overlay and recomputation stores
        self.last_mesh   = None
        self.last_result = None

        # MediaPipeLandmarker configuration
        base_options = mp_python.BaseOptions(
            model_asset_path=landmarker_model
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True # Enabled to support head pose estimation
        )

        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def compute_eye_aspect_ratio(self, p1, p2, p3, p4, p5, p6):
        # Apply the eye aspect ratio formula on provided landmarks
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        v3 = np.linalg.norm(p1 - p4)

        return (v1 + v2) / (2.0 * v3)

    def eye_aspect_ratio_handler(self, mesh):
        # Point transformation to convert normalized landmark to pixel coordinates
        pt = lambda i: np.array([mesh[i].x, mesh[i].y])

        # Determine eye-specific landmark points
        left_eye_ar  = self.compute_eye_aspect_ratio(pt(33), pt(160), pt(158), pt(133), pt(153), pt(144))
        right_eye_ar = self.compute_eye_aspect_ratio(pt(362), pt(385), pt(387), pt(263), pt(373), pt(380))

        # Calculate the average eye aspect ratio to improve stability
        averaged_ar = (left_eye_ar + right_eye_ar) / 2.0

        return averaged_ar

    def get_head_pose_ratios(self, mesh):
        # Horizontal edge markers
        nose_x     = mesh[1].x       # Nose tip
        left_edge  = mesh[234].x     # Left face edge
        right_edge = mesh[454].x     # Right face edge

        # Vertical edge markers
        nose_y        = mesh[1].y    # Nose tip
        forehead_edge = mesh[10].y   # Top of forehead
        chin_edge     = mesh[152].y  # Bottom of chin

        face_width = right_edge - left_edge
        face_height = chin_edge - forehead_edge

        if face_width == 0 or face_height == 0:
            return None

        horizontal_ratio = (nose_x - left_edge)     / face_width
        vertical_ratio   = (nose_y - forehead_edge) / face_height

        return horizontal_ratio, vertical_ratio
    
    def get_head_euler_angles(self, result):
        if not result.facial_transformation_matrixes:
            return None
    
        mp_matrix  = result.facial_transformation_matrixes[0]
        rtn_matrix = np.array(mp_matrix)[:3, :3]

        pitch = np.degrees(np.arctan2(-rtn_matrix[1, 2], rtn_matrix[2, 2]))
        yaw   = np.degrees(np.arctan2(rtn_matrix[0, 2],  np.sqrt(rtn_matrix[1, 2] ** 2) + rtn_matrix[2, 2] **2 ))

        return yaw, pitch

    def draw_overlay(self, display_frame):
        if self.last_mesh is None:
            return display_frame

        display_frame_height, display_frame_width, _ = display_frame.shape

        # Draw a box around the detected face
        x_coordinates = [int(landmark.x * display_frame_width)  for landmark in self.last_mesh]
        y_coordinates = [int(landmark.y * display_frame_height) for landmark in self.last_mesh]

        x_left, x_right = min(x_coordinates), max(x_coordinates)
        y_bottom, y_top = min(y_coordinates), max(y_coordinates)

        cv2.rectangle(display_frame, (x_left, y_bottom), (x_right, y_top), (0, 255, 0), 2)

        # Draw a face mask using the landmarks extracted with MediaPipe
        for landmark in self.last_mesh:
            landmark_x = int(landmark.x * display_frame_width)
            landmark_y = int(landmark.y * display_frame_height)
            cv2.circle(display_frame, (landmark_x, landmark_y), 1, (255, 255, 255), -1)

        return display_frame

    def process_frame(self, frame):
        # Use the input frame as the input image for MediaPipe
        image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self.landmarker.detect(image)

        eye_aspect_ratio = 0.0
        if not results.face_landmarks:
            return "LOST", eye_aspect_ratio, 0.0, 0.0

        mesh = results.face_landmarks[0]
        self.last_mesh = mesh

        euler = self.get_head_euler_angles(results)
        if euler is None:
            return "LOST", 0.0, 0.0, 0.0

        current_yaw, current_pitch = euler
        eye_aspect_ratio = self.eye_aspect_ratio_handler(mesh)

        return "CALIBRATING", eye_aspect_ratio, current_yaw, current_pitch
    

def setup_logger(log_level):
    """
    Setup logger with the specified log level.

    :param log_level: The logging level to use in logger configuration
    """

    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True)


def setup_argument_parser():
    """
    Setup argument parser for command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-ll",  "--log_level",            help="Logging level to use with logging library", default="INFO", choices=LOG_OPTIONS)
    parser.add_argument("-fps", "--target_fps",           help="Target FPS for video processing", default=TARGET_FPS, type=int)
    parser.add_argument("-lm",  "--landmarker_model",     help="Path to MediaPipe face landmarker task model", default=LANDMARK_PREDICTION_MODEL)
    parser.add_argument("-s",   "--show_simple",          help="Show landmarks on a black canvas instead of raw video", action="store_true")
    parser.add_argument("-e",   "--ear_threshold",        help="Eye aspect ratio threshold for detecting drowsiness", default=DEFAULT_EAR_THRESHOLD, type=float)
    parser.add_argument("-o",   "--observation_window",   help="Observation safety window in seconds", default=OBSERVATION_SAFETY_WINDOW, type=float)
    parser.add_argument("-c",   "--calibration_duration", help="Target duration for eye threshold calibration process", default=CALIBRATION_DURATION, type=float)
    parser.add_argument("-da",  "--disable_annotation",   help="Disable annotation on video output", action="store_true")
    parser.add_argument("-DA",  "--disable_api",          help="Disable FastAPI integration", action="store_true")
    parser.add_argument("-l",   "--log",                  help="Enable csv logging", action="store_true")

    return parser.parse_args()


def check_observation_status(head_direction, observation_status, window):
    """
    Check if the driver has completed precautionary observations within a window

    :param head_direction     : The direction the driver's head is facing
    :param observation_status : A dictionary of timestamps when head direction was recorded
    :param window             : The time window (in seconds) to consider for observation safety
    """

    current_time = time.time()

    if head_direction == "LEFT":
        observation_status["left"] = current_time
    elif head_direction == "RIGHT":
        observation_status["right"] = current_time

    left_look_completed  = (current_time - observation_status["left"]) < window
    right_look_completed = (current_time - observation_status["right"]) < window

    return (left_look_completed and right_look_completed)


def annotate_video(frame, video_width, video_height, video_type, fps, eye_ar, ear_threshold, head_direction, observation_complete):
    """
    Annotate the video stream with resolution and frame rate information.

    :param frame                : captured frame
    :param video_width          : width of the video stream
    :param video_height         : height of the video stream
    :param video_type           : type of video capture method
    :param fps                  : frame rate of video stream
    :param eye_ar               : eye aspect ratio
    :param ear_threshold        : eye aspect ratio threshold for determining driver status
    :param gaze_direction       : direction of driver's gaze
    :param observation_complete : status of the driver observation
    """

    # Blend overlay into original frame to achieve semi-transparent effect
    overlay = frame.copy()

    panel_height = 100
    left_panel_width, right_panel_width = 240, 260
    alpha, beta = 0.7, 0.3

    cv2.rectangle(overlay, (0, 0), (left_panel_width, panel_height), (0, 0, 0), -1)                           # System stats
    cv2.rectangle(overlay, (video_width - right_panel_width, 0), (video_width, panel_height), (0, 0, 0), -1)  # Driver status
    cv2.addWeighted(overlay, alpha, frame, beta, 0, frame)

    # Text Strings
    resolution_str = f"Resolution: {video_width}x{video_height}"
    fps_str        = f"FPS: {int(fps)}"
    eye_ar_str     = f"Eye Aspect Ratio: {eye_ar:.2f}"
    head_str       = f"Head Direction: {head_direction}"
    video_type_str = f"Camera: {video_type}"

    # Styling Characteristics
    thickness = 1
    scale     = 0.5
    font      = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA  # Anti-aliasing

    if head_direction == "LOST":
        eye_status_str   = "FACE LOST"
        eye_status_color = (0, 165, 255)
    elif eye_ar < ear_threshold:
        eye_status_str = "EYES CLOSED"
        eye_status_color = (0, 0, 255)
    else:
        eye_status_str = "EYES OPEN"
        eye_status_color = (0, 255, 0)

    if observation_complete:
        obs_status_str = "OBSERVATION COMPLETE"
        obs_status_color = (0, 255, 0)
    else:
        obs_status_str = "OBSERVATION INCOMPLETE"
        obs_status_color = (0, 0, 255)

    # [LEFT-ALIGNED] System stats
    cv2.putText(frame, fps_str,        (10, 60),  font, scale, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, resolution_str, (10, 30),  font, scale, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, eye_ar_str,     (10, 90),  font, scale, (200, 200, 200),  thickness, line_type)

    # [RIGHT-ALIGNED] Driver status
    cv2.putText(frame, eye_status_str, (video_width - 175, 60),  font, scale, eye_status_color, thickness, line_type)
    cv2.putText(frame, head_str,       (video_width - 240, 30),  font, scale, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, obs_status_str, (video_width - 240, 90),  font, scale, obs_status_color, thickness, line_type)

    # [BOTTOM-ALIGNED] Video stats
    cv2.putText(frame, video_type_str, (video_width - 175, video_height - 15), font, scale, (200, 200, 200), thickness, line_type)


def main():
    logging.info("Starting VideoDetectionModule-RAS...")

    args = setup_argument_parser()
    setup_logger(args.log_level)

    # Initialize classes
    video_stream = CameraStream(index=CAMERA_INDEX)
    video_logger = VideoLogger(active=args.log)
    calibrator   = CalibrationManager(duration=args.calibration_duration)
    detector     = LandmarkDetector(landmarker_model=args.landmarker_model)

    # The program should not continue if the camera failed to initialize
    if not video_stream.isOperational():
        logging.error("Camera failed to operate")
        exit()

    if not args.disable_api:
        logging.info(f"Starting API server at http://{HOST_NAME}:{PORT_NUMBER}/status...")

        # Keep daemon set to true to force API thread to run in background
        api_server_thread = Thread(target=start_api,  daemon=True)
        api_server_thread.start()

    target_fps = args.target_fps
    frame_duration = 1.0 / target_fps

    video_width, video_height = video_stream.get_dimensions()
    video_type = video_stream.get_camera_type()

    # Observation maintenance variables
    last_known_head_direction = "FORWARD"
    observation_status        = {"left": 0, "right": 0}
    observation_timestamp     = time.time()
    current_ear_threshold     = args.ear_threshold

    # Detection optimization variables
    frame_count = 0

    while(True):
        start_time = time.perf_counter()

        # Continuously capture a frame with success/failure return value
        ret, raw_frame = video_stream.read()

        if not ret or raw_frame is None or raw_frame.size == 0:
            logging.warning("Frame capture failed, skipping frame...")
            continue

        frame_flipped    = cv2.flip(raw_frame, 1)
        frame_downscaled = cv2.resize(frame_flipped, (DOWNSCALED_FRAME_WIDTH_PX, DOWNSCALED_FRAME_HEIGHT_PX))
        frame            = cv2.cvtColor(frame_downscaled, cv2.COLOR_BGR2RGB)

        if frame_count % DETECTION_SKIPPED_FRAMES == 0:
            # Use processed ratio and pose values to get predicted head direction
            status, eye_aspect_ratio, raw_yaw, raw_pitch = detector.process_frame(frame)
            head_direction = calibrator.get_informed_direction(status, raw_yaw, raw_pitch)
            detector.last_result = (head_direction, eye_aspect_ratio)
        else:
            # Use stored ratio and pose values on non-detection frames
            if detector.last_result is not None:
                head_direction, eye_aspect_ratio = detector.last_result
            else:
                head_direction, eye_aspect_ratio = "LOST", 0.0

        if head_direction == "LOST":
            # Use the last known head direction within a short timeout to reduce noise between head pose transitions
            if (time.time() - observation_timestamp) < OBSERVATION_LOST_TIMEOUT:
                head_direction = last_known_head_direction
                eye_ar = eye_aspect_ratio
            else:
                eye_ar = 0.0
        else:
            eye_ar = eye_aspect_ratio
            last_known_head_direction = head_direction
            observation_timestamp = time.time()
            detector.last_result = (head_direction, eye_aspect_ratio)

        observation_complete = check_observation_status(head_direction, observation_status, args.observation_window)

        if not args.disable_api:
            update_status(eye_ar, head_direction, observation_complete, current_ear_threshold)

        # Synchronize loop speed with target fps, providing software capped frame rate
        processing_time = time.perf_counter() - start_time

        sleep_duration = frame_duration - processing_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)

        total_loop_time = time.perf_counter() - start_time

        fps = 1.0 / total_loop_time

        if args.show_simple:
            output_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            output_frame = frame_flipped

        detector.draw_overlay(output_frame)

        # Perform eye aspect ratio threshold calibration
        if not calibrator.is_calibration_done:
            calibrator.update_calibration(head_direction, eye_aspect_ratio, yaw=raw_yaw, pitch=raw_pitch)
            current_ear_threshold = calibrator.threshold

            # Annotate calibration progress on flipped frame
            cv2.putText(output_frame, calibrator.get_progress_string(), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, cv2.LINE_AA)

        # Add text stating system, driver, and video statistics
        if not args.disable_annotation:
            annotate_video(output_frame, video_width, video_height, video_type, fps, eye_ar, current_ear_threshold, head_direction,
                           observation_complete)

        # Start recording video stream data logs
        if args.log and calibrator.is_calibration_done:
            video_logger.record(eye_ar, current_ear_threshold, head_direction, observation_complete)

        cv2.imshow('VideoDetectionModule - RAS', output_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"Target: {target_fps} FPS | Actual: {fps:.2f} FPS | Processing Time: {total_loop_time:.4f}s")

        if cv2.waitKey(1) == ord('q'):
            break

    # Close the video stream and corresponding windows after usage
    video_stream.release()
    cv2.destroyAllWindows()

    if args.log:
        video_logger.flush_buffer()
        logging.info(f"Logs successfully saved to {video_logger.filename}")


if __name__ == "__main__":
    main()
