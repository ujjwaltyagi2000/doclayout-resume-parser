import os
import pandas as pd

OUTPUT_DIR = "sections_output_2105"  # change this
FINAL_FILE = os.path.join(OUTPUT_DIR, "original_prompt_results_2105.csv")

def merge_csv_files():
    all_dfs = []

    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
    csv_files.sort()  # ensure correct order (batch_1, batch_2, ...)

    if not csv_files:
        print("❌ No CSV files found in OUTPUT_DIR")
        return

    for file in csv_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        print(f"📄 Reading: {file}")

        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")

    if not all_dfs:
        print("❌ No valid CSVs to merge")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    merged_df.to_csv(FINAL_FILE, index=False)
    print(f"✅ Merged file saved at: {FINAL_FILE}")


if __name__ == "__main__":
    merge_csv_files()