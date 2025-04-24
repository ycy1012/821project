import os
import shutil
import pandas as pd
from cli import run_pipline

def test_run_pipeline_partial_input():
    # Setup
    input_dir = "input_images"
    output_dir = "tests/output"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "test_features.csv")

    # Select just a few sample images to test
    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])[:2]
    assert len(sample_files) > 0, "No sample files found for testing."

    sample_input_dir = os.path.join(output_dir, "sample_input")
    os.makedirs(sample_input_dir, exist_ok=True)

    # Copy sample files to temporary input folder
    for fname in sample_files:
        shutil.copy(os.path.join(input_dir, fname), os.path.join(sample_input_dir, fname))

    # Run pipeline on sample input
    run_pipline.run_pipeline(sample_input_dir, output_csv)

    # Assertions
    assert os.path.exists(output_csv), "Output CSV was not created."
    df = pd.read_csv(output_csv)
    assert df.shape[0] == len(sample_files), "Mismatch in number of feature rows."
    assert df.shape[1] == 8, "Feature vector should have 7 features + filename."

    # Clean up
    shutil.rmtree(output_dir)
