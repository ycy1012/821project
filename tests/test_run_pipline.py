import os
import shutil
import pandas as pd
from cli import run_pipline


def prepare_sample_input(sample_dir, input_dir, num_files=2):
    os.makedirs(sample_dir, exist_ok=True)
    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])[
        :num_files
    ]
    for fname in sample_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(sample_dir, fname)
        shutil.copy(src, dst)
    return sample_files


def validate_csv(csv_path, expected_files, expected_feat_dim):
    assert os.path.exists(csv_path), f"{csv_path} was not created."
    df = pd.read_csv(csv_path)
    assert df.shape[0] == len(expected_files), f"{csv_path}: row count mismatch."
    assert df.shape[1] == expected_feat_dim + 1, (
        f"{csv_path}: expected {expected_feat_dim} features + filename."
    )
    assert list(df["filename"]) == expected_files, (
        f"{csv_path}: filenames do not match."
    )


def test_run_pipeline_handcrafted_and_deep():
    input_dir = "input_images"
    output_dir = "tests/output"
    sample_input_dir = os.path.join(output_dir, "sample_input")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare input
    sample_files = prepare_sample_input(sample_input_dir, input_dir)

    # Run CLI for both feature types
    run_pipline.extract_and_save(
        "handcrafted",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_handcrafted.csv"),
    )
    run_pipline.extract_and_save(
        "deep",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_deep.csv"),
    )

    # Validate handcrafted
    validate_csv(
        os.path.join(output_dir, "features_handcrafted.csv"),
        sample_files,
        expected_feat_dim=7,
    )

    # Validate deep
    validate_csv(
        os.path.join(output_dir, "features_deep.csv"),
        sample_files,
        expected_feat_dim=512,
    )

    # Optional: cleanup
    shutil.rmtree(output_dir)
