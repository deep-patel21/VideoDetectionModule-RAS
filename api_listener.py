import os
import time
import requests
import argparse

from videodetectionmode_api import PORT_NUMBER

INTERVAL = 0.02  # in seconds [Translates to 50hz]
TIMEOUT  = 3     # in seconds


def setup_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-hn", "--host_name",   help="hostname to run API server",  default="localhost",   type=str)
    parser.add_argument("-pn", "--port_number", help="port to run API server",      default=PORT_NUMBER, type=int)
    parser.add_argument("-i",  "--interval",    help="Polling interval in seconds", default=INTERVAL,    type=float)
    
    return parser.parse_args()


def format_status(status, elapsed):
    timestamp            = status.get("timestamp", 0)
    driver_status        = status.get("driver_status",  "UNKNOWN")
    head_direction       = status.get("head_direction", "UNKNOWN")
    observation_complete = status.get("observation_complete",  False)

    driver_color      = "\033[92m" if driver_status == "ALERT" else "\033[91m"
    head_color        = "\033[93m" if head_direction not in ("FORWARD", "LOST") else "\033[97m"
    observation_color = "\033[92m" if observation_complete else "\033[91m"
    reset             = "\033[0m"

    observation_str = "COMPLETE" if observation_complete else "INCOMPLETE"

    return (
        f"\n"
        f"  \033[1mVideoDetectionModule-RAS - API Listener\033[0m\n"
        f"         =======================================\n"
        f"  Timestamp      : {time.strftime('%H:%M:%S', time.localtime(timestamp))}\n"
        f"  Driver Status  : {driver_color}{driver_status}{reset}\n"
        f"  Head Direction : {head_color}{head_direction}{reset}\n"
        f"  Observation    : {observation_color}{observation_str}{reset}\n"
        f"         =======================================\n"
        f"  Polling every {elapsed * 1000:.0f}ms  |  Press Ctrl+C to stop\n"
    )


def main():
    args    = setup_argument_parser()

    url      = f"http://{args.host_name}:{args.port_number}/status"
    attempts = 0

    print(f"  Connecting to {url}...")

    while True:
        start = time.time()

        # Request the API endpoint to retrieve VideoDetectionModule data
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            status   = response.json()

            # Clear the terminal and display the updated data
            if os.name == "nt":     # Windows
                os.system('cls')
            else:                   # Other
                os.system('clear')
            
            print(format_status(status, args.interval))
        except requests.exceptions.ConnectionError:
            attempts += 1
            print(f"\nConnection failed on attempt {attempts}")
        except requests.exceptions.Timeout:
            print(f"\nRequest timed out after {TIMEOUT} seconds.")
        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting API Listener")