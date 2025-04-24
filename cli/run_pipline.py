import argparse
import os
import pandas as pd
from preprocessing.preprocess import preprocess_image
from features.handcrafted import extract_handcrafted_features
from features.deep import extract_deep_features


def extract_and_save(features_type: str, image_paths: list, output_path: str):
    results = []

    for path in image_paths:
        try:
            img = preprocess_image(path)
            if features_type == "handcrafted":
                feat = extract_handcrafted_features(img)
            elif features_type == "deep":
                feat = extract_deep_features(img)
            results.append([os.path.basename(path)] + feat.tolist())
        except Exception as e:
            print(f"[WARNING] Failed on {path}: {e}")

    df = pd.DataFrame(
        results,
        columns=["filename"] + [f"feat_{i}" for i in range(len(results[0]) - 1)],
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved {features_type} features to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ChestX-FeatLib CLI")
    parser.add_argument("--input", required=True, help="Path to input image folder")
    parser.add_argument(
        "--output", required=True, help="Output directory to save feature CSVs"
    )
    parser.add_argument(
        "--features",
        choices=["handcrafted", "deep", "all"],
        default="handcrafted",
        help="Feature type to extract",
    )

    args = parser.parse_args()

    image_paths = [
        os.path.join(args.input, f)
        for f in sorted(os.listdir(args.input))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if args.features in ["handcrafted", "all"]:
        extract_and_save(
            "handcrafted",
            image_paths,
            os.path.join(args.output, "features_handcrafted.csv"),
        )
    if args.features in ["deep", "all"]:
        extract_and_save(
            "deep", image_paths, os.path.join(args.output, "features_deep.csv")
        )


if __name__ == "__main__":
    main()
