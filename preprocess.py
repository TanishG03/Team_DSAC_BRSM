import os
import pandas as pd
 
# Root dataset folder
DATA_ROOT = "data"
 
# Output folder
OUTPUT_ROOT = "preprocessed_data"
 
 
def ensure_folder(path):
    os.makedirs(path, exist_ok=True)
 
 
def preprocess_single_phone():
    input_folder = os.path.join(DATA_ROOT, "single", "phone")
    output_folder = os.path.join(OUTPUT_ROOT, "single", "phone")
    ensure_folder(output_folder)
 
    columns_to_drop = [
        'PlayerID'
    ]
 
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
 
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
 
            df = pd.read_csv(input_path)
 
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
 
            df.to_csv(output_path, index=False)
 
    print("Finished preprocessing single/phone")
 
 
def preprocess_single_lab_manual():
 
    keep_columns = [
        'target_col',
        'trial.started',
        'mouse.started',
        'square.started',
        'target.started',
        'trial.stopped',
        'mouse.x',
        'mouse.y',
        'mouse.time',
    ]
 
    input_folder = os.path.join(DATA_ROOT, "single", "lab")
    output_folder = os.path.join(OUTPUT_ROOT, "single", "lab")
    ensure_folder(output_folder)
 
    for fname in os.listdir(input_folder):
 
        if not fname.endswith(".csv"):
            continue
 
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)
 
        df = pd.read_csv(input_path)
 
        cols_to_keep = [c for c in keep_columns if c in df.columns]
 
        df_kept = df[cols_to_keep]
 
        df_kept.to_csv(output_path, index=False)
 
    print("Finished preprocessing single/lab")
 
 
def preprocess_multiple_phone():
 
    input_folder = os.path.join(DATA_ROOT, "multiple", "phone")
    output_folder = os.path.join(OUTPUT_ROOT, "multiple", "phone")
    ensure_folder(output_folder)
 
    columns_to_drop = [
        # add columns here if needed
        'PlayerID'
    ]
 
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
 
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
 
            df = pd.read_csv(input_path)
 
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
 
            df.to_csv(output_path, index=False)
 
    print("Finished preprocessing multiple/phone")
 
 
def preprocess_multiple_lab_manual():
 
    keep_columns = [
        'target_col',
        'trial.started',
        'square.started',
        'mouse.started',
        'target.started',
        'trial.stopped',
        'click_times',
        'mouse.x',
        'mouse.y',
        'mouse.time',
    ]
 
    input_folder = os.path.join(DATA_ROOT, "multiple", "lab")
    output_folder = os.path.join(OUTPUT_ROOT, "multiple", "lab")
    ensure_folder(output_folder)
 
    for fname in os.listdir(input_folder):
 
        if not fname.endswith(".csv"):
            continue
 
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)
 
        df = pd.read_csv(input_path)
 
        cols_to_keep = [c for c in keep_columns if c in df.columns]
 
        df_kept = df[cols_to_keep]
 
        df_kept.to_csv(output_path, index=False)
 
    print("Finished preprocessing multiple/lab")
 
 
def main():
 
    preprocess_single_phone()
    preprocess_single_lab_manual()
    preprocess_multiple_phone()
    preprocess_multiple_lab_manual()
 
 
if __name__ == "__main__":
    main()