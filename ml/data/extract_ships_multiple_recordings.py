import os
import pandas as pd
import shutil

ships = pd.read_csv('image_segmentation/data/ships_with_multiple_recordings.csv')
cargo_ships = ships[ships["class"] == "Cargo"]["ship_name"].to_list()
passenger_ships = ships[ships["class"] == "Passengership"]["ship_name"].to_list()
tanker_ships = ships[ships["class"] == "Tanker"]["ship_name"].to_list()
tug_ships = ships[ships["class"] == "Tug"]["ship_name"].to_list()

SRC = 'deepship_baseline_unnorm_mat/'
DEST = 'deepship_baseline_unnorm_multi/'

for class_folder in os.listdir(SRC):
    class_path = os.path.join(SRC, class_folder)

    if not os.path.isdir(class_path):
        continue

    for f in os.listdir(class_path):
        if not f.endswith('.mat'):
            continue
            
        ship_name = f.split('-')[0]

        if class_folder == "Cargo":
            if ship_name not in cargo_ships:
                continue
        elif class_folder == "Passengership":
            if ship_name not in passenger_ships:
                continue
        elif class_folder == "Tanker":
            if ship_name not in tanker_ships:
                continue
        elif class_folder == "Tug":
            if ship_name not in tug_ships:
                continue
        
        original_path = os.path.join(SRC, class_folder, f)
        new_path = os.path.join(DEST, class_folder, f)
        shutil.copy2(original_path, new_path)