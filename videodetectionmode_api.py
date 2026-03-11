import time
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

HOST_NAME = "0.0.0.0"
PORT_NUMBER = 8000


app = FastAPI(title="VideoDetectionModule-RAS API")


class DriverStatus(BaseModel):
    timestamp            : float
    driver_status        : str
    head_direction       : str
    observation_complete : bool


current_status = DriverStatus(
    timestamp            = 0.0,
    driver_status        = "UNKNOWN",
    head_direction       = "LOST",
    observation_complete = False,
)


@app.get("/status", response_model=DriverStatus)
def get_videodetectionmodule_status():
    """
    API endpoint to retrieve the latest driver statuses
    """

    return current_status


def update_status(eye_ar, head_direction, observation_complete, ear_threshold):
    """
    Update the shared DriverStatus object read by the FastAPI endpoint.

    :param eye_ar               : eye aspect ratio
    :param head_direction       : direction of driver's gaze
    :param observation_complete : status of the driver observation
    :param ear_threshold        : eye aspect ratio threshold for determining driver status
    """

    global current_status

    current_status = DriverStatus(
        timestamp            = time.time(),
        driver_status        = "ALERT" if eye_ar >= ear_threshold else "DROWSY",
        head_direction       = head_direction,
        observation_complete = observation_complete,
    )


def start_api(port=PORT_NUMBER):
    """
    Launch the API server using uvicorn.
    """
    uvicorn.run(app, host=HOST_NAME, port=port, log_level="warning")


if __name__ == "__main__":
    start_api()