import os
import time
import json

STORAGE_FOLDER = "json_statuses"
JSON_FILE = os.path.join(STORAGE_FOLDER, "video_status.json")

print(f"[INFO] Listening to {JSON_FILE}...")

timestamp_prev = 0

try:
    while True:
        if os.path.exists(JSON_FILE):
            retries = 5

            for x in range(retries):
                try:
                    with open(JSON_FILE, "r") as file:
                        data = json.load(file)

                        if data['timestamp'] != timestamp_prev:
                            
                            print(f"[{time.strftime('%H:%M:%S')}] "
                                f"Status: {data['driver_status']} | "
                                f"Head: {data['head_direction']} | "
                                f"Obs: {data['observation_complete']}")
                            
                            timestamp_prev = data['timestamp']
                    break
                except (PermissionError, json.JSONDecodeError):
                    if x < retries - 1:
                        time.sleep(0.05)
                    else:
                        continue
        else:
            print(f"[WARNING] {JSON_FILE} not found.")

        time.sleep(0.5)
except KeyboardInterrupt:
    print("\n[INFO] Exiting listening session.")