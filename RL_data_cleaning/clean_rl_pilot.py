"""
Author: Kat Seitz (katharinaseitz2029@u.northwestern.edu)
Date: 2025-06-13
Description:
    This script processes trial-level data from behavioral runs.
"""

import os
import pandas as pd
import re
from datetime import datetime
import glob
import numpy as np

def extract_datetime(filename):
    """Extract datetime from filename for sorting."""
    # Extract date and time from filename
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}h\d{2}\.\d{2}\.\d{3})', filename)
    if match:
        date_str, time_str = match.groups()
        # Convert to datetime object for sorting
        time_str = time_str.replace('h', ':')
        datetime_str = f"{date_str} {time_str}"
        return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M.%S.%f')
    return None

def process_subject_files(subject_dir):
    """Process all CSV files for a single subject."""
    # Find all non-practice CSV files
    csv_files = glob.glob(os.path.join(subject_dir, "*out-of-scanner*.csv"))
    
    # Skip practice files
    csv_files = [f for f in csv_files if "practice" not in f.lower()]
    
    if not csv_files:
        print(f"No files found for {subject_dir}")
        return None
    
    # Sort files by datetime
    csv_files.sort(key=extract_datetime)
    
    all_runs = []
    
    for iter_idx, file_path in enumerate(csv_files, 1):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if this is a valid data file with trials
            if 'trials.thisN' not in df.columns or df['trials.thisN'].dropna().empty:
                print(f"Skipping {file_path} - no trial data found")
                continue
            
            # Extract relevant columns
            # For choice, we need to determine if the participant chose left or right
            # Based on the header inspection, cue_resp.keys contains 'j' or 'k' for left/right
            
            run_data = pd.DataFrame()
            
            # Get trial indices - need to filter out rows that don't have trial data
            valid_trial_indices = df['trials.thisN'].dropna().index
            
            # Trial information
            run_data['trial_number'] = df.loc[valid_trial_indices, 'trials.thisN'].reset_index(drop=True)
            run_data['run_number'] = (run_data['trial_number'] == 0).cumsum() + (iter_idx - 1)*3
            
            # Choice (binary): j=left (0), k=right (1)
            sides = df.loc[valid_trial_indices, 'trials.cue_resp.keys'].reset_index(drop=True)
            run_data['side_picked'] = sides.map(lambda x: 0 if x == 'j' else 1 if x == 'k' else None)
            
            # What side was better?
            run_data['good_side'] = df.loc[valid_trial_indices, 'good_side'].reset_index(drop=True)       
            run_data['good_side'] = df.loc[valid_trial_indices, 'good_side'].reset_index(drop=True).map({'left': 0, 'right': 1}).astype(float)

            # Did they pick the better side?
            run_data['accuracy'] = np.where(
                run_data['side_picked'].isna(),
                np.nan,
                (run_data['side_picked'] == run_data['good_side']).astype(int))
  
            
            # Were they rewarded for their choice?
            outcomes = df.loc[valid_trial_indices, 'outcome_image'].reset_index(drop=True)
            run_data['outcome'] = outcomes.map(lambda x: 1 if 'coin' in str(x) else 0 if 'empty' in str(x) else None)
            
            # Reaction time
            run_data['reaction_time'] = df.loc[valid_trial_indices, 'trials.cue_resp.rt'].reset_index(drop=True)
            
            all_runs.append(run_data)
            
            print(f"Processed {file_path} - Found {len(run_data)} trials")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_runs:
        print(f"No valid data processed for {subject_dir}")
        return None
    
    # Concatenate all runs
    subject_data = pd.concat(all_runs, ignore_index=True)
    
    return subject_data

def main():
    # TODO change as needed
    # Base directory containing all subjects' data
    base_dirs = [
        os.path.join('/Users/katharinaseitz/Desktop/RL_data_cleaning/raw_data/NU'),
        os.path.join('/Users/katharinaseitz/Desktop/RL_data_cleaning/raw_data/UCB')
    ]
    # TODO change as needed
    # Create output directory if it doesn't exist
    output_dir = os.path.join('/Users/katharinaseitz/Desktop/RL_data_cleaning/processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    for base_dir in base_dirs:
        # Find all subject directories
        subject_dirs = [d for d in glob.glob(os.path.join(base_dir, "*_Pilot_*")) 
                       if os.path.isdir(d)]
        
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            print(f"Processing {subject_id}...")
            
            # Process all files for this subject
            subject_data = process_subject_files(subject_dir)
            
            if subject_data is not None:
                # Save concatenated data to CSV
                output_file = os.path.join(output_dir, f"{subject_id}_concatenated.csv")
                subject_data.to_csv(output_file, index=False)
                print(f"Saved {len(subject_data)} trials to {output_file}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()