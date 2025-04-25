import os
import shutil
import pandas as pd
from cli import run_pipline
import pytest

def prepare_sample_input(sample_dir, input_dir, num_files=5):
    """
    Prepare a sample subset of input images for testing.

    Args:
        sample_dir (str): Directory to save the sample images.
        input_dir (str): Directory where the original input images are stored.
        num_files (int): Number of sample images to copy. Default is 5.

    Returns:
        list: Filenames of the copied sample images.
    """
    os.makedirs(sample_dir, exist_ok=True)
    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])[:num_files]
    for fname in sample_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(sample_dir, fname)
        shutil.copy(src, dst)
    return sample_files

def validate_csv(csv_path, expected_files, expected_feat_dim):
    """
    Validate that a CSV file contains the expected number of rows and columns.

    Args:
        csv_path (str): Path to the generated CSV file.
        expected_files (list): List of filenames that should be present.
        expected_feat_dim (int): Number of expected features (excluding filename).

    Raises:
        AssertionError: If the CSV structure does not match expectations.
    """
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
    """
    Test that both handcrafted and deep feature CSVs are generated correctly.
    """
    input_dir = "input_images"
    output_dir = "tests/output"
    sample_input_dir = os.path.join(output_dir, "sample_input")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare sample input images
    sample_files = prepare_sample_input(sample_input_dir, input_dir)

    # Extract and save handcrafted features
    run_pipline.extract_and_save(
        "handcrafted",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_handcrafted.csv"),
    )

    # Extract and save deep features
    run_pipline.extract_and_save(
        "deep",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_deep.csv"),
    )

    # Validate the generated handcrafted features CSV
    validate_csv(
        os.path.join(output_dir, "features_handcrafted.csv"),
        sample_files,
        expected_feat_dim=7,
    )

    # Validate the generated deep features CSV
    validate_csv(
        os.path.join(output_dir, "features_deep.csv"),
        sample_files,
        expected_feat_dim=512,
    )

    # Clean up output files after test
    shutil.rmtree(output_dir)

def test_run_pipeline_only_handcrafted():
    """
    Test that only handcrafted features are extracted and saved correctly.
    """
    input_dir = "input_images"
    output_dir = "tests/output_handcrafted_only"
    sample_input_dir = os.path.join(output_dir, "sample_input")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare sample input images
    sample_files = prepare_sample_input(sample_input_dir, input_dir)

    # Extract and save handcrafted features only
    run_pipline.extract_and_save(
        "handcrafted",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_handcrafted.csv"),
    )

    # Validate the generated handcrafted features CSV
    validate_csv(
        os.path.join(output_dir, "features_handcrafted.csv"),
        sample_files,
        expected_feat_dim=7,
    )

    shutil.rmtree(output_dir)

def test_run_pipeline_only_deep():
    """
    Test that only deep features are extracted and saved correctly.
    """
    input_dir = "input_images"
    output_dir = "tests/output_deep_only"
    sample_input_dir = os.path.join(output_dir, "sample_input")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare sample input images
    sample_files = prepare_sample_input(sample_input_dir, input_dir)

    # Extract and save deep features only
    run_pipline.extract_and_save(
        "deep",
        [os.path.join(sample_input_dir, f) for f in sample_files],
        os.path.join(output_dir, "features_deep.csv"),
    )

    # Validate the generated deep features CSV
    validate_csv(
        os.path.join(output_dir, "features_deep.csv"),
        sample_files,
        expected_feat_dim=512,
    )

    shutil.rmtree(output_dir)

def test_extract_and_save_runtimeerror_on_empty_input():
    """
    Test that RuntimeError is raised when input images are invalid or missing.
    """
    bad_image_path = "nonexistent_folder/nonexistent_image.png"
    output_csv = "tests/output/invalid_output.csv"

    # Expect a RuntimeError due to no valid features extracted
    with pytest.raises(RuntimeError, match="No features extracted"):
        run_pipline.extract_and_save(
            features_type="handcrafted",
            image_paths=[bad_image_path],
            output_path=output_csv
        )
