import argparse
import os
import pandas as pd
from preprocessing.preprocess import preprocess_image
from features.handcrafted import extract_handcrafted_features

def run_pipeline(input_folder: str, output_csv: str):
    results = []

    for fname in sorted(os.listdir(input_folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(input_folder, fname)
            try:
                preprocessed_img = preprocess_image(fpath)
                features = extract_handcrafted_features(preprocessed_img)
                results.append([fname] + features.tolist())
            except Exception as e:
                print(f"[WARNING] Failed on {fname}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results, columns=["filename"] + [f"feat_{i}" for i in range(7)])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="ChestX-FeatLib CLI")
    parser.add_argument("--input", required=True, help="Path to input image folder")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    parser.add_argument("--features", choices=["handcrafted"], default="handcrafted", help="Type of features to extract")

    args = parser.parse_args()

    if args.features == "handcrafted":
        run_pipeline(args.input, args.output)

if __name__ == "__main__":
    main()
