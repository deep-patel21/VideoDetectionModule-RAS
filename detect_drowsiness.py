import cv2
import time
import argparse
import logging
import numpy as np

from threading import Thread
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance


LOG_OPTIONS     = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
RISK_SEVERITIES = ["MILD", "MODERATE", "SEVERE"]

CAMERA_INDEX    = 0
TARGET_FPS      = 24


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

    parser.add_argument("-ll", "--log_level", help="Logging level to use with logging library", default="INFO", choices=LOG_OPTIONS)
    parser.add_argument("-v",  "--target_fps", help="Target FPS for video processing", default=TARGET_FPS, type=int)

    return parser.parse_args()


def compute_eye_aspect_ratio():
    """
    Compute the eye aspect ratio using Euclidean distances.
    """

    pass


def get_video_dimensions(video_stream):
    """
    Return the width and height of the video stream
    
    :param video_stream: instance of cv2 VideoCapture
    """

    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def annotate_video(frame, video_width, video_height, fps):
    """
    Annotate the video stream with resolution and frame rate information.

    :param frame       : captured frame
    :param video_width : width of the video stream
    :param video_height: height of the video stream
    :param fps         : frame rate of video stream
    """

    resolution_str = f"Resolution: {video_width}x{video_height}"
    fps_str        = f"FPS: {int(fps)}"
    
    cv2.putText(frame, resolution_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, fps_str,        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


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

    while(True):
        start_time = time.time()

        # Continuosly capture a frame with success/failure return value
        ret, frame = video_stream.read()

        if not ret: 
            logging.warning("Encountered frame drop. Exiting video stream.")
            break

        # Synchronize loop speed with target fps, providing software capped frame rate
        processing_time = time.time() - start_time

        if processing_time < frame_duration:
            time.sleep(frame_duration - processing_time)

        # Compute real-time fps of the video stream
        fps = 1.0 / (time.time() - start_time)

        # Add text stating resolution and frames per second of video capture
        annotate_video(frame, video_width, video_height, fps)

        frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('VideoDetectionModule - RAS', frame_greyscale)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # Close the video stream and corresponding windows after usage
    video_stream.release()
    cv2.destroyAllWindows()        

if __name__ == "__main__":
    main()
