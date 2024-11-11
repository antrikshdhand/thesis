import os
import pandas as pd
import random

NPZ = True # Toggle between true for npz and false for csv

base_path = 'deepship_ps_2_csv'

# A dictionary of SHIP_NAME: {RECORDING_DATE: NUM_SEG, *[RECORDING_DATE: NUM_SEG, ...]}
ships = {}

# Collect all files with their full paths
all_files = []
for class_folder in os.listdir(base_path):
    class_path = os.path.join(base_path, class_folder)
    
    if os.path.isdir(class_path):
        for file_name in os.listdir(class_path):
            if not file_name.endswith('.csv'):
                continue

            split = file_name.split('-')

            # Collect ship recording information
            ship_name = split[0]
            date_seg = split[2].split('.')[0]
            date = date_seg.split('_')[0]
            seg = int(date_seg.split('_')[1].lstrip('seg'))
            if ship_name not in ships:
                ships[ship_name] = {}
            else:
                ships[ship_name][date] = seg
            
            if NPZ:
                class_path = os.path.join('deepship_ps_2_npz', class_folder)

            file_path = os.path.join("data", class_path, file_name)
            
            # Change the file extension if needed
            if NPZ:
                file_path = file_path.rsplit('.', 1)[0] + '.npz'
            
            all_files.append((ship_name, file_path, date, seg))

# Only those ships with multiple recordings
ships_multiple_recordings = {}
for key, val in ships.items():
    if len(val) > 1:
        ships_multiple_recordings[key] = val

all_files_df = pd.DataFrame(all_files, columns=['ship_name', 'file_path', 'date', 'seg'])
all_files_df = all_files_df.sort_values(['ship_name', 'file_path']).reset_index(drop=True)
multiple_recordings_df = all_files_df[all_files_df["ship_name"].isin(ships_multiple_recordings)]

print(multiple_recordings_df) # A dataframe containing all the files related to ships with more than 1 recording.

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

    # Iterate over each segment within the grouped ship data
    for _, row in ship_data.iterrows():

        # Skip this segment if it has already been paired
        if row["file_path"] in used_files:
            continue

        # Get all recordings for the same ship that are not from the same date
        current_date = row['date']
        other_recordings = ship_data[ship_data['date'] != current_date]

        if other_recordings.empty:
            # Skip if no other recording dates available
            continue
        
        # Randomly select a segment from other recordings
        random_pair = other_recordings.sample(1).iloc[0]
        
        # Skip if the randomly selected segment has already been paired
        if random_pair["file_path"] in used_files:
            continue

        # Add the selected pair to the pairings dictionary
        pairings["ship_name"].append(ship_name)
        pairings["file_path_1"].append(row["file_path"])
        pairings["date_seg_1"].append(f"{row['date']}_{row['seg']}")
        pairings["file_path_2"].append(random_pair["file_path"])
        pairings["date_seg_2"].append(f"{random_pair['date']}_{random_pair['seg']}")

        # Mark both segments as used
        used_files.add(row["file_path"])
        used_files.add(random_pair["file_path"])

pairings_df = pd.DataFrame(pairings)
# print(pairings_df)
pairings_df.to_csv("deepship_pairs_diff_recording.csv", index=False)

# # Group by ship name and create pairs
# pairs = []
# for ship_name, group in df.groupby('ship_name'):
#     file_paths = group['file_path'].tolist()
    
#     for i in range(0, len(file_paths) - 1, 2):
#         pairs.append((file_paths[i], file_paths[i + 1]))

# pairs_df = pd.DataFrame(pairs, columns=['input', 'label'])
# pairs_df.to_csv('deepship_pairs_2.csv', index=False)
