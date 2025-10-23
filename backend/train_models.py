"""
Training script for PhishNet ML models.
"""
import os
import sys
from data.dataset_generator import generate_synthetic_dataset, save_dataset, load_dataset
from models.ml_models import PhishingDetector


def train_phishnet_models():
    """Train all PhishNet models on the dataset."""
    print("PhishNet Model Training")
    print("=" * 50)
    
    # Generate or load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'phishing_dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        X, y = load_dataset(dataset_path)
    else:
        print("Generating synthetic dataset...")
        X, y = generate_synthetic_dataset(n_samples=1000)
        save_dataset(X, y, dataset_path)
        print(f"Dataset saved to {dataset_path}")
    
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


if __name__ == "__main__":
    try:
        detector, scores = train_phishnet_models()
        print("\n" + "=" * 50)
        print("Training completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)
