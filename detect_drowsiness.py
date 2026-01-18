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

    while(True):
        pass

if __name__ == "__main__":
    main()
