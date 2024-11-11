import os
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

def convert_csvs_to_npzs(csv_root_dir: str, npzs_dir: str):
    """
    Converts a folder `csv_root_dir` containing subfolders with CSV files 
    into a new folder `npzs_dir` containing subfolders with npz files.

    :param csv_root_dir: Path to the root directory containing subfolders
        containing csv files.
    :param npzs_dir: Path to the root directory where subfolders containing
        npz files will be stored.
    """
    # Check if the CSV root directory exists
    if not os.path.exists(csv_root_dir):
        print(f"""CSV root directory '{csv_root_dir}' not found. 
              Make sure you are running this script from the correct
              directory.""", file=sys.stderr)
        return

    # Create the root folder for npz files if it doesn't exist
    os.makedirs(npzs_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(csv_root_dir) if os.path.isdir(os.path.join(csv_root_dir, d))]
    for class_dir in tqdm(class_dirs, desc="Class folders", unit="folder"):
        class_path = os.path.join(csv_root_dir, class_dir)

        # Make the new class subdirectory in the npzs dir
        npzs_class_path = os.path.join(npzs_dir, class_dir)
        os.makedirs(npzs_class_path, exist_ok=True)

        # List all CSV files in the current class directory
        csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]

        # Convert each CSV file to npz 
        for f in tqdm(csv_files, desc=f"Converting {class_dir} files", leave=False, unit="file"):
            csv_file_path = os.path.join(class_path, f)
            np_data = pd.read_csv(csv_file_path, header=None).values 

            npz_file = os.path.splitext(f)[0] + '.npz' 
            npz_file_path = os.path.join(npzs_class_path, npz_file)
            
            # Save as npz (compressed)
            np.savez_compressed(npz_file_path, np_data=np_data)

def main():
    convert_csvs_to_npzs('deepship_ps_2_csv', 'deepship_ps_2_npz')

if __name__ == '__main__':
    main()