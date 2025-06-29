import pandas as pd
import os

def parquet_to_csv(parquet_file_path: str, csv_file_path: str):
    try:
        if not os.path.exists(parquet_file_path):
            print(f"Error: Parquet file not found at '{parquet_file_path}'")
            return

        df = pd.read_parquet(parquet_file_path)

        output_directory = os.path.dirname(csv_file_path)
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)

        df.to_csv(csv_file_path, index=False)
        print(f"CSV file saved to: {csv_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

input_parquet_file = "master_longitudinal_table.parquet"  
output_csv_file = "output.csv"     

parquet_to_csv(input_parquet_file, output_csv_file)
