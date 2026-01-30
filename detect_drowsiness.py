import os
import cv2
import dlib
import time
import argparse
import logging
import numpy as np

from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance


LANDMARK_PREDICTION_MODEL = "models/shape_predictor_68_face_landmarks.dat"

(LEFT_START, LEFT_END)   = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(RIGHT_START, RIGHT_END) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

LOG_OPTIONS     = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
RISK_SEVERITIES = ["MILD", "MODERATE", "SEVERE"]

CAMERA_INDEX              = 0
TARGET_FPS                = 24
EAR_THRESHOLD             = 0.17
OBSERVATION_SAFETY_WINDOW = 5.0     # [seconds]
OBSERVATION_LOST_TIMEOUT  = 0.8

CLAHE_LIMIT_NO_BOOST    = 1.0
CLAHE_LIMIT_LOW_BOOST   = 1.5
CLAHE_LIMIT_MID_BOOST   = 2.0
CLAHE_LIMIT_HIGH_BOOST  = 2.5
CLAHE_LIMIT_ULTRA_BOOST = 3.0
CLAHE_LIMIT_EXTRM_BOOST = 3.5
CLAHE_LIMIT_MAX_BOOST   = 4.0


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

    parser.add_argument("-ll",  "--log_level",          help="Logging level to use with logging library", default="INFO", choices=LOG_OPTIONS)
    parser.add_argument("-fps", "--target_fps",         help="Target FPS for video processing", default=TARGET_FPS, type=int)
    parser.add_argument("-m",   "--dlib_model",         help="Path to the Dlib landmark prediction model", default=LANDMARK_PREDICTION_MODEL, type=str)
    parser.add_argument("-s",   "--show_simple",        help="Show landmarks on a black canvas instead of raw video", action="store_true")
    parser.add_argument("-e",   "--ear_threshold",      help="Eye aspect ratio threshold for detecting drowsiness", default=EAR_THRESHOLD, type=float)
    parser.add_argument("-o",   "--observation_window", help="Observation safety window in seconds", default=OBSERVATION_SAFETY_WINDOW, type=float)
    parser.add_argument("-da",  "--disable_annotation", help="Disable annotation on video output", action="store_true")

    return parser.parse_args()


def setup_facial_recognition(dlib_model_path):
    """
    Return dlib facial detection and prediction instances

    :param dlib_model_path: Path to the Dlib landmark prediction model
    """

    if not os.path.exists(dlib_model_path):
        logging.error(f"Landmark prediction model not found at {dlib_model_path}")
        exit()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_model_path)

    return detector, predictor


def compute_eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio using Euclidean distances.

    :param eye: singular eye landmark set
    """

    v1 = distance.euclidean(eye[1], eye[5])
    v2 = distance.euclidean(eye[2], eye[4])
    v3 = distance.euclidean(eye[0], eye[3])

    return (v1 + v2) / (2.0 * v3)


def eye_aspect_ratio_handler(shape):
    """
    Compute the eye aspect ratio using Euclidean distances.

    :param shape: facial landmarks extracted from the face
    """

    # Determine eye specific landmark points
    left_eye  = shape[LEFT_START:LEFT_END]
    right_eye = shape[RIGHT_START:RIGHT_END]

    left_eye_ar  = compute_eye_aspect_ratio(left_eye)
    right_eye_ar = compute_eye_aspect_ratio(right_eye)

    # Calculate the average eye aspect ratio to improve stability
    averaged_ar = (left_eye_ar + right_eye_ar) / 2.0

    return averaged_ar


def head_pose_handler(shape):
    """
    Compute the head pose direction based on facial landmarks.

    :param shape: facial landmarks extracted from the face
    """

    # Identify key landmarks corresponding to face symmetry
    nose_bridge = shape[27][0]
    left_edge   = shape[0][0]
    right_edge  = shape[16][0]

    face_width = right_edge - left_edge

    if face_width == 0:
        return "FORWARD"

    central_ratio = (nose_bridge - left_edge) / face_width

    if central_ratio < 0.40:
        return "LEFT"
    elif central_ratio > 0.63:
        return "RIGHT"
    else:
        return "FORWARD"


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

    left_look_completed = (current_time - observation_status["left"]) < window
    right_look_completed = (current_time - observation_status["right"]) < window

    return (left_look_completed and right_look_completed)


def get_video_dimensions(video_stream):
    """
    Return the width and height of the video stream

    :param video_stream: instance of cv2 VideoCapture
    """

    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def adaptive_clahe(frame_greyscale):
    """
    Apply adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) to a greyscale frame.

    :param frame_greyscale : frame captured in video stream converted to COLOR_BGR2GRAY
    """

    avg_brightness = np.mean(frame_greyscale)

    if avg_brightness > 180:
        clip_limit = CLAHE_LIMIT_NO_BOOST     # Well lit environment
    elif avg_brightness > 150:
        clip_limit = CLAHE_LIMIT_LOW_BOOST
    elif avg_brightness > 120:
        clip_limit = CLAHE_LIMIT_MID_BOOST    # Moderately lit environment
    elif avg_brightness > 90:
        clip_limit = CLAHE_LIMIT_HIGH_BOOST
    elif avg_brightness > 60:
        clip_limit = CLAHE_LIMIT_ULTRA_BOOST
    elif avg_brightness > 40:
        clip_limit = CLAHE_LIMIT_EXTRM_BOOST
    else:
        clip_limit = CLAHE_LIMIT_MAX_BOOST   # Dim environment

    clahe_model = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe_model.apply(frame_greyscale), clip_limit


def annotate_video(frame, video_width, video_height, fps, eye_ar, ear_threshold, head_direction, observation_complete, active_clip_limit):
    """
    Annotate the video stream with resolution and frame rate information.

    :param frame                : captured frame
    :param video_width          : width of the video stream
    :param video_height         : height of the video stream
    :param fps                  : frame rate of video stream
    :param eye_ar               : eye aspect ratio
    :param gaze_direction       : direction of driver's gaze
    :param observation_complete : status of the driver observation
    :param active_clip_limit    : currently applied histogram equalization clip limit
    """

    # Blend overlay into original frame to achieve semi-transparent effect
    overlay = frame.copy()

    panel_height = 135
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
    clip_str       = f"CLAHE Clip Limit: {active_clip_limit}"

    # Styling Characteristics
    thickness = 1
    scale     = 0.6
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
    cv2.putText(frame, fps_str,        (10, 60),  font, 0.5, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, resolution_str, (10, 30),  font, 0.5, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, eye_ar_str,     (10, 90),  font, 0.5, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, clip_str,       (10, 120), font, 0.5, (200, 200, 200),  thickness, line_type)

    # [RIGHT-ALIGNED] Driver status
    cv2.putText(frame, eye_status_str, (video_width - 175, 60),  font, 0.5, eye_status_color, thickness, line_type)
    cv2.putText(frame, head_str,       (video_width - 240, 30),  font, 0.5, (200, 200, 200),  thickness, line_type)
    cv2.putText(frame, obs_status_str, (video_width - 240, 90),  font, 0.5, obs_status_color, thickness, line_type)


def main():
    args = setup_argument_parser()
    setup_logger(args.log_level)

    video_stream = cv2.VideoCapture(CAMERA_INDEX)

    target_fps = args.target_fps
    frame_duration = 1 / target_fps

    # The program should not continue if the webcam failed to initialize
    if not video_stream.isOpened():
        logging.error("Camera failed to operate")
        exit()

    video_width, video_height = get_video_dimensions(video_stream)
    detector, predictor = setup_facial_recognition(args.dlib_model)

    last_known_head_direction = "FORWARD"
    observation_status = {"left": 0, "right": 0}
    observation_timestamp = time.time()

    while(True):
        start_time = time.time()

        # Continuosly capture a frame with success/failure return value
        ret, raw_frame = video_stream.read()
        frame = cv2.flip(raw_frame, 1)

        # Tracking metrics are reset each iteration
        eye_ar = 0
        head_direction = "NONE"

        if not ret:
            logging.warning("Encountered frame drop. Exiting video stream.")
            break

        # Detect facial landmarks on histogram equalized frame
        frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_greyscale, active_clip_limit = adaptive_clahe(frame_greyscale)

        if args.show_simple:
            display_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            display_frame = cv2.cvtColor(frame_greyscale, cv2.COLOR_GRAY2BGR)

        observation_complete = False

        faces = detector(frame_greyscale, 0)

        if len(faces) > 0:
            observation_timestamp = time.time()

            for face in faces:
                x_left, x_right = max(0, face.left()), max(0, face.right())
                y_top, y_bottom = max(0, face.top()), max(0, face.bottom())

                # Draw a box around the detected face
                cv2.rectangle(display_frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

                shape = predictor(frame_greyscale, face)
                shape = face_utils.shape_to_np(shape)

                eye_ar = eye_aspect_ratio_handler(shape)
                head_direction = head_pose_handler(shape)
                last_known_head_direction = head_direction

                observation_complete = check_observation_status(head_direction, observation_status, args.observation_window)

                # Draw a face mask using the 68-landmarks extracted with dlib
                for (x, y) in shape:
                    cv2.circle(display_frame, (x, y), 1, (255, 255, 255), -1)
        else:
            if (time.time() - observation_timestamp) < OBSERVATION_LOST_TIMEOUT:
                head_direction = last_known_head_direction
                observation_complete = check_observation_status(head_direction, observation_status, args.observation_window)
            else:
                head_direction = "LOST"

        # Synchronize loop speed with target fps, providing software capped frame rate
        processing_time = time.time() - start_time

        if processing_time < frame_duration:
            time.sleep(frame_duration - processing_time)

        fps = 1.0 / (time.time() - start_time)

        # Add text stating resolution and frames per second of video capture
        if not args.disable_annotation:
            annotate_video(display_frame, video_width, video_height, fps, eye_ar, args.ear_threshold, head_direction,
                           observation_complete, active_clip_limit)

        cv2.imshow('VideoDetectionModule - RAS', display_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Close the video stream and corresponding windows after usage
    video_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
