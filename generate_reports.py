import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

STORAGE_FOLDER = "csv_logs"

def setup_argument_parser():
    """
    Setup argument parser for command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-f",  "--file_target", help="File target to use for report generation")

    return parser.parse_args()


def get_latest_csv_file():
    """
    Get the most recent CSV file from the storage folder.
    """

    csv_files = glob.glob(f"{STORAGE_FOLDER}/*.csv")

    if not csv_files:
        print(f"[INFO] No CSV files found in {STORAGE_FOLDER}.")
        return None

    latest_file = max(csv_files, key=os.path.getctime)

    return latest_file


def report_eye_aspect_ratio(ax1, df):
    """
    Create a plot of the eye aspect ratio over time
    """

    ax1.set_title("Eye Aspect Ratio v.s. Time", fontsize=14)
    ax1.set_xlabel("Session Time (seconds)")
    ax1.set_ylabel("Eye Aspect Ratio")

    # Apply rolling average to reduce noise
    df['eye_ar_smooth'] = df['eye_ar'].rolling(window=7, center=True).mean()
    ax1.plot(df['seconds'], df['eye_ar_smooth'].interpolate(), color='Black', linewidth=1, label='Eye Aspect Ratio')
    ax1.axhline(y=0.17, color='r', linestyle='--', alpha=0.6, label='Eyes Closed Threshold (0.17)')

    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')


def report_observation_status(ax2, df):
    """
    Create a plot of the observation status and validity over time
    """

    ax2.set_title("Observation Status v.s. Time", fontsize=14)
    ax2.set_ylabel("Observation Status")
    ax2.set_xlabel("Session Time (seconds)")

    direction_map = {"LEFT": -1, "FORWARD": 0, "RIGHT": 1, "LOST": 0}
    df['head_index'] = df['head_direction'].map(direction_map)

    # Apply rolling average to reduce noise
    df['head_index_smooth'] = df['head_index'].rolling(window=5, center=True).apply(
        lambda x: pd.Series(x).value_counts().idxmax()
    )

    ax2.step(df['seconds'], df['head_index_smooth'], where='post', color='Black',
             linewidth=1.5, label='Head Direction', alpha=0.8)

    # Report lost statuses as red dots
    lost_status = df[df['head_direction'] == "LOST"]
    ax2.scatter(lost_status['seconds'], lost_status['head_index'],
                color='red', s=10, label='Face Lost', zorder=5)

    ax2.fill_between(df['seconds'], -1.2, 1.2,
                 where=(df['observation_complete'] == 1),
                 color='#27ae60', alpha=0.2, step='post', label='Observation Valid')

    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['LEFT', 'FORWARD', 'RIGHT'])
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='best')


def main(file_path=None):

    # Filter logs for most recent session data
    if file_path:
        if not os.path.exists(file_path):
            print(f"[ERROR] Specified file not found: {file_path}")
            return
        target_file = file_path
    else:
        target_file = get_latest_csv_file()
        if not target_file:
            return

    # Extract data from latest csv file
    print(f"[INFO] Generating report from {target_file}...")
    df = pd.read_csv(target_file)
    df['seconds'] = df['timestamp'] - df['timestamp'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title(f'VideoDetectionModule-RAS Report: {os.path.basename(target_file)}')
    plt.subplots_adjust(hspace=0.3)

    report_eye_aspect_ratio(ax1, df)
    report_observation_status(ax2, df)

    if not os.path.exists("reports"):
        os.makedirs("reports")

    base_name = os.path.basename(target_file)
    report_name = base_name.replace(".csv", ".png")
    report_path = os.path.join("reports", report_name)

    plt.savefig(report_path)
    print(f"[SUCCESS] Report saved to: {report_path}")

    plt.show()


if __name__ == "__main__":
    args = setup_argument_parser()
    main(file_path=args.file_target)
