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


def setup_logger(log_level):
    """Setup logger with the specified log level."""

    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    logging.basicConfig(level=numeric_level, 
                        format="%(asctime)s [%(levelname)s] %(message)s", 
                        force=True)


def setup_argument_parser():
    """Setup argument parser for command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-ll", "--log_level", help="Logging level to use with logging library", default="INFO", choices=LOG_OPTIONS)

    return parser.parse_args()


def compute_eye_aspect_ratio():
    """Compute the eye aspect ratio using Euclidean distances."""
    pass


def main():
    args = setup_argument_parser()
    setup_logger(args.log_level)

    video_stream = cv2.VideoCapture(CAMERA_INDEX)

    # The program should not continue if the webcam failed to initialize
    if not video_stream:
        logging.error("Camera failed to operate")
        exit()

    while(True):
        # Continuosly capture a frame with success/failure return value
        ret, frame = video_stream.read()

        if not ret: 
            logging.warning("Encountered frame drop. Exiting video stream.")
            break

        frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame', frame_greyscale)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # Close the video stream and corresponding windows after usage
    video_stream.release()
    cv2.destroyAllWindows()        

if __name__ == "__main__":
    main()
