"""
Training script for PhishNet ML models.
"""
import os
import sys
import argparse
import warnings
import pandas as pd
from data.dataset_generator import generate_synthetic_dataset, save_dataset, load_dataset
from models.ml_models import PhishingDetector


def train_phishnet_models():
    """Train all PhishNet models on the dataset.

    This function now looks for CSV files under `backend/data/` and will
    concatenate all files that contain an `is_phishing` column. If none are
    found, it falls back to the existing single-file behavior and synthetic
    generation.
    """
    print("PhishNet Model Training")
    print("=" * 50)

    # Default data directory (can be overridden via CLI)
    default_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Discover CSV files in data directory
    csv_files = []
    try:
        for fname in os.listdir(default_data_dir):
            if fname.lower().endswith('.csv'):
                csv_files.append(os.path.join(default_data_dir, fname))
    except Exception:
        csv_files = []

    # Load and concatenate CSVs that contain 'is_phishing' column
    datasets = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            if 'is_phishing' in df.columns:
                print(f"Including dataset: {path} ({len(df)} rows)")
                datasets.append(df)
            else:
                warnings.warn(f"Skipping {path}: missing 'is_phishing' column")
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")

    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        y = combined['is_phishing'].values
        X = combined.drop('is_phishing', axis=1)
        print(f"Loaded combined dataset with {len(X)} samples from {len(datasets)} file(s)")
    else:
        # Fallback to original single-file behavior
        dataset_path = os.path.join(default_data_dir, 'phishing_dataset.csv')
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            X, y = load_dataset(dataset_path)
        else:
            print("No CSV datasets found with 'is_phishing' column. Generating synthetic dataset...")
            X, y = generate_synthetic_dataset(n_samples=1000)
            # Save a copy so users see a dataset file
            try:
                save_dataset(X, y, dataset_path)
                print(f"Dataset saved to {dataset_path}")
            except Exception:
                warnings.warn(f"Could not save synthetic dataset to {dataset_path}")
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Legitimate samples: {sum(y == 0)}")
    print(f"Phishing samples: {sum(y == 1)}")
    print(f"Features: {list(X.columns)}")
    
    # Initialize detector
    print("\nInitializing PhishingDetector...")
    detector = PhishingDetector()
    
    # Train models
    print("\nTraining models...")
    scores = detector.train(X.values, y)
    
    print("\nTraining completed!")
    print("\nModel Accuracies:")
    print("-" * 50)
    for model_name, accuracy in scores.items():
        print(f"{model_name.replace('_', ' ').title()}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save models
    print("\nSaving trained models...")
    detector.save_models()
    print("Models saved successfully!")
    
    return detector, scores


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhishNet models")
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Optional directory containing CSV dataset files (overrides default)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # If user provided a data-dir, temporarily adjust the data directory search
    if args.data_dir:
        # Monkey-patch default data dir by changing environment variable used by this script
        # Simpler: change current working directory when searching; pass in via global variable would be heavier
        # We'll set the expected data directory path by replacing the folder used inside the function.
        # Easiest approach: change os.path.dirname(__file__) reference by chdir; do it carefully.
        try:
            os.chdir(os.path.dirname(__file__))
        except Exception:
            pass
        # If provided path is absolute, use it; otherwise join to script dir
        if os.path.isabs(args.data_dir):
            data_dir = args.data_dir
        else:
            data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
        # Temporarily rename default_data_dir variable by injecting CSVs into expected folder path
        # We'll create a small wrapper to call train_phishnet_models while temporarily pointing os.listdir
        original_listdir = os.listdir
        def _listdir_override(path):
            if path == os.path.join(os.path.dirname(__file__), 'data'):
                try:
                    return original_listdir(data_dir)
                except Exception:
                    return []
            return original_listdir(path)
        os.listdir = _listdir_override

    try:
        detector, scores = train_phishnet_models()
        print("\n" + "=" * 50)
        print("Training completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)
    finally:
        # Restore os.listdir if we overrode it
        try:
            if 'original_listdir' in locals():
                os.listdir = original_listdir
        except Exception:
            pass
