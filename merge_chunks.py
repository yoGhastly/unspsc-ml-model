import pandas as pd
import os

def merge_formatted_chunks(chunk_folder, output_file):
    """
    Merges all formatted chunk CSV files into a single CSV file.
    
    Args:
        chunk_folder (str): Directory where the formatted chunk files are stored.
        output_file (str): Path to the final merged CSV file.
    
    Returns:
        None
    """
    # List all formatted chunk files in the specified folder
    chunk_files = [os.path.join(chunk_folder, f) for f in os.listdir(chunk_folder) if f.endswith('_formatted.csv')]
    
    # Check if there are any chunk files to process
    if not chunk_files:
        print("No formatted chunk files found in the specified folder.")
        return
    
    # Initialize a list to hold DataFrames
    dfs = []
    
    # Read each chunk file into a DataFrame and append it to the list
    for chunk_file in chunk_files:
        print(f"Reading {chunk_file}...")
        df = pd.read_csv(chunk_file)
        dfs.append(df)
    
    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save the final merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    
    print(f"All chunks merged into '{output_file}'.")

# Parameters
chunk_folder = 'formatted_chunks/'  # Folder where the formatted chunks are stored
output_file = 'merged_output.csv'  # Path to the final merged CSV file

# Merge formatted chunks
merge_formatted_chunks(chunk_folder, output_file)
