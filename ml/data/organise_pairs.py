import os
import pandas as pd
import numpy as np
import random
from typing import Literal

def get_dataset_info(path_to_root: str, ext: Literal['csv', 'npz', 'mat']):
    """
    Collects metadata for each file in the dataset directory.

    :param path_to_root: Root directory containing the dataset.
    :param ext: File extension to filter (e.g., 'mat', 'csv', 'npz').
    :return: 
        - A pd.DataFrame with columns (ship_name, file_path, date, seg).
        - A dictionary tracking the number of unique recordings per vessel.
    """
    ships = {} # Dictionary of {ship_name: {recording_date: num_segments}}
    all_files = []

    for class_folder in os.listdir(path_to_root):
        class_path = os.path.join(path_to_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            if not file_name.endswith(ext):
                continue

            ship_name, _, date_seg_ext = file_name.split('-')
            date, seg_ext = date_seg_ext.split('_')
            seg = int(seg_ext.split('.')[0].lstrip('seg'))

            if ship_name not in ships:
                ships[ship_name] = {}
            ships[ship_name][date] = seg
            
            file_path = os.path.join(class_path, file_name)
            all_files.append((ship_name, class_folder, file_path, date, seg))

    all_files_df = pd.DataFrame(all_files, columns=['ship_name', 'class', 'file_path', 'date', 'seg'])
    return all_files_df, ships 

def make_pairs_different_recording(multiple_recordings_df: pd.DataFrame):
    """
    Creates pairs of recordings from different dates for each ship.

    :param multiple_recordings_df: DataFrame of ships with multiple recordings.
    :return: DataFrame with pairs of recordings from different dates.
    """
    pairings = {
        "ship_name": [],
        "file_path_1": [],
        "date_seg_1": [],
        "file_path_2": [],
        "date_seg_2": []
    }
    used_files = set()
    random.seed(42)

    for ship_name, ship_data in multiple_recordings_df.groupby("ship_name"):
        for _, row in ship_data.iterrows():
            if row["file_path"] in used_files:
                continue

            current_date = row['date']
            other_recordings = ship_data[ship_data['date'] != current_date]

            if other_recordings.empty:
                continue
            
            random_pair = other_recordings.sample(1).iloc[0]
            if random_pair["file_path"] in used_files:
                continue

            pairings["ship_name"].append(ship_name)
            pairings["file_path_1"].append(row["file_path"])
            pairings["date_seg_1"].append(f"{row['date']}_{row['seg']}")
            pairings["file_path_2"].append(random_pair["file_path"])
            pairings["date_seg_2"].append(f"{random_pair['date']}_{random_pair['seg']}")

            used_files.update([row["file_path"], random_pair["file_path"]])

    pairings_df = pd.DataFrame(pairings)
    pairings_df.to_csv("deepship_pairs_diff_recording.csv", index=False)
    return pairings_df

def make_pairs_same_recording(all_files_df: pd.DataFrame):
    """
    Creates pairs of recordings from the same date for each ship.

    :param all_files_df: DataFrame containing all files metadata.
    :return: DataFrame with pairs of recordings from the same date.
    """
    pairs = []
    for ship_name, group in all_files_df.groupby('ship_name'):
        file_paths = group['file_path'].tolist()
        for i in range(0, len(file_paths) - 1, 2):
            pairs.append((file_paths[i], file_paths[i + 1]))

    pairs_df = pd.DataFrame(pairs, columns=['input', 'label'])
    pairs_df.to_csv('deepship_pairs_same_recording.csv', index=False)
    return pairs_df

def main(path_to_root: str, ext: Literal['csv', 'npz', 'mat']):
    """
    Main function to create pairs of recordings for the dataset.

    :param path_to_root: Root directory containing the dataset.
    :param ext: File extension of the recordings.
    """
    all_files_df, ships = get_dataset_info(path_to_root, ext)

    # Filter ships which have multiple recordings
    ships_multiple_recordings = {k: v for k, v in ships.items() if len(v) > 1}
    multiple_recordings_df = all_files_df[all_files_df["ship_name"].isin(ships_multiple_recordings)]

    ships_with_multiple_recordings = multiple_recordings_df[["ship_name", "class"]].drop_duplicates()
    ships_with_multiple_recordings.to_csv("ships_with_multiple_recordings.csv", index=False)

    # Create pairs from different recordings
    diff_recordings_pairs = make_pairs_different_recording(multiple_recordings_df)

    # Create pairs from the same recording
    same_recording_pairs = make_pairs_same_recording(all_files_df)

    print("Pairs from different recordings:")
    print(diff_recordings_pairs.head())
    print("\nPairs from the same recording:")
    print(same_recording_pairs.head())

if __name__ == "__main__":
    main(path_to_root="deepship_baseline_mat", ext="mat")