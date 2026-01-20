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

CAMERA_INDEX    = 0
TARGET_FPS      = 24
EAR_THRESHOLD   = 0.15


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

    parser.add_argument("-ll", "--log_level",     help="Logging level to use with logging library", default="INFO", choices=LOG_OPTIONS)
    parser.add_argument("-v",  "--target_fps",    help="Target FPS for video processing", default=TARGET_FPS, type=int)
    parser.add_argument("-m",  "--dlib_model",    help="Path to the Dlib landmark prediction model", default=LANDMARK_PREDICTION_MODEL, type=str)
    parser.add_argument("-s",  "--show_simple",   help="Show landmarks on a black canvas instead of raw video", action="store_true")
    parser.add_argument("-e",  "--ear_threshold", help="Eye aspect ratio threshold for detecting drowsiness", default=EAR_THRESHOLD, type=float)

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


def get_video_dimensions(video_stream):
    """
    Return the width and height of the video stream

    :param video_stream: instance of cv2 VideoCapture
    """

    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def annotate_video(frame, video_width, video_height, fps, eye_ar, ear_threshold):
    """
    Annotate the video stream with resolution and frame rate information.

    :param frame        : captured frame
    :param video_width  : width of the video stream
    :param video_height : height of the video stream
    :param fps          : frame rate of video stream
    :param eye_ar       : eye aspect ratio
    """

    resolution_str = f"Resolution: {video_width}x{video_height}"
    fps_str        = f"FPS: {int(fps)}"
    eye_ar_str     = f"Eye Aspect Ratio: {eye_ar:.2f}"

    if eye_ar < ear_threshold:
        eye_status_str = "EYES CLOSED"
        eye_status_color = (0, 0, 255)
    else:
        eye_status_str = "EYES OPEN"
        eye_status_color = (0, 255, 0)  

    # Draw annotation elements
    cv2.putText(frame, resolution_str,                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),  2)
    cv2.putText(frame, fps_str,                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),  2)
    cv2.putText(frame, eye_ar_str,                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),  2)
    cv2.putText(frame, eye_status_str, (video_width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_status_color, 2)


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

    while(True):
        start_time = time.time()

        # Continuosly capture a frame with success/failure return value
        ret, raw_frame = video_stream.read()
        frame = cv2.flip(raw_frame, 1)
        
        # Eye aspect ratio is initally set to 0
        eye_ar = 0

        if not ret:
            logging.warning("Encountered frame drop. Exiting video stream.")
            break

        # Detect facial landmarks
        frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(frame_greyscale, 0)

        if args.show_simple:
            display_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            display_frame = cv2.cvtColor(frame_greyscale, cv2.COLOR_GRAY2BGR)

        for face in faces:
            x_left, x_right = max(0, face.left()), max(0, face.right())
            y_top, y_bottom = max(0, face.top()), max(0, face.bottom())

            # Draw a box around the detected face
            cv2.rectangle(display_frame, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

            shape = predictor(frame_greyscale, face)
            shape = face_utils.shape_to_np(shape)

            eye_ar = eye_aspect_ratio_handler(shape)

            # Draw a face mask using the 68-landmarks extracted with dlib
            for (x, y) in shape:
                cv2.circle(display_frame, (x, y), 1, (255, 255, 255), -1)

        # Synchronize loop speed with target fps, providing software capped frame rate
        processing_time = time.time() - start_time

        if processing_time < frame_duration:
            time.sleep(frame_duration - processing_time)

        # Compute real-time fps of the video stream
        fps = 1.0 / (time.time() - start_time)

        # Add text stating resolution and frames per second of video capture
        annotate_video(display_frame, video_width, video_height, fps, eye_ar, args.ear_threshold)
        cv2.imshow('VideoDetectionModule - RAS', display_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Close the video stream and corresponding windows after usage
    video_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
