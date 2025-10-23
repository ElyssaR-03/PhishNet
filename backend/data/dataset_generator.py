"""
Generate synthetic phishing dataset for training.
In a real application, this would be replaced with actual phishing data.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def generate_synthetic_dataset(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a synthetic dataset for phishing detection.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features_dataframe, labels)
    """
    np.random.seed(42)
    
    # Generate half legitimate, half phishing
    n_legitimate = n_samples // 2
    n_phishing = n_samples - n_legitimate
    
    # Legitimate URL features (generally shorter, fewer suspicious patterns)
    legitimate_data = {
        'url_length': np.random.normal(30, 10, n_legitimate).clip(10, 100),
        'num_dots': np.random.poisson(2, n_legitimate),
        'num_hyphens': np.random.poisson(0.5, n_legitimate),
        'num_underscores': np.random.poisson(0.3, n_legitimate),
        'num_slashes': np.random.poisson(3, n_legitimate),
        'num_questions': np.random.binomial(1, 0.2, n_legitimate),
        'num_equals': np.random.poisson(0.5, n_legitimate),
        'num_at': np.zeros(n_legitimate),
        'num_ampersands': np.random.poisson(0.3, n_legitimate),
        'num_digits': np.random.poisson(2, n_legitimate),
        'has_ip': np.zeros(n_legitimate),
        'is_https': np.random.binomial(1, 0.8, n_legitimate),
        'domain_length': np.random.normal(15, 5, n_legitimate).clip(5, 40),
        'path_length': np.random.normal(10, 5, n_legitimate).clip(0, 50),
        'query_length': np.random.exponential(5, n_legitimate).clip(0, 30),
        'has_suspicious_keywords': np.random.binomial(1, 0.1, n_legitimate),
    }
    
    # Phishing URL features (longer, more suspicious patterns)
    phishing_data = {
        'url_length': np.random.normal(60, 20, n_phishing).clip(20, 200),
        'num_dots': np.random.poisson(4, n_phishing),
        'num_hyphens': np.random.poisson(2, n_phishing),
        'num_underscores': np.random.poisson(1.5, n_phishing),
        'num_slashes': np.random.poisson(5, n_phishing),
        'num_questions': np.random.binomial(1, 0.5, n_phishing),
        'num_equals': np.random.poisson(2, n_phishing),
        'num_at': np.random.binomial(1, 0.3, n_phishing),
        'num_ampersands': np.random.poisson(1.5, n_phishing),
        'num_digits': np.random.poisson(5, n_phishing),
        'has_ip': np.random.binomial(1, 0.3, n_phishing),
        'is_https': np.random.binomial(1, 0.5, n_phishing),
        'domain_length': np.random.normal(25, 10, n_phishing).clip(10, 60),
        'path_length': np.random.normal(25, 10, n_phishing).clip(5, 80),
        'query_length': np.random.exponential(15, n_phishing).clip(0, 60),
        'has_suspicious_keywords': np.random.binomial(1, 0.7, n_phishing),
    }
    
    # Combine data
    df_legitimate = pd.DataFrame(legitimate_data)
    df_phishing = pd.DataFrame(phishing_data)
    
    # Create labels (0 = legitimate, 1 = phishing)
    y_legitimate = np.zeros(n_legitimate)
    y_phishing = np.ones(n_phishing)
    
    # Combine and shuffle
    X = pd.concat([df_legitimate, df_phishing], ignore_index=True)
    y = np.concatenate([y_legitimate, y_phishing])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X.iloc[indices].reset_index(drop=True)
    y = y[indices]
    
    return X, y


def save_dataset(X: pd.DataFrame, y: np.ndarray, filepath: str) -> None:
    """Save dataset to CSV file."""
    df = X.copy()
    df['is_phishing'] = y
    df.to_csv(filepath, index=False)


def load_dataset(filepath: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    y = df['is_phishing'].values
    X = df.drop('is_phishing', axis=1)
    return X, y


if __name__ == "__main__":
    # Generate and save a dataset
    X, y = generate_synthetic_dataset(1000)
    save_dataset(X, y, "phishing_dataset.csv")
    print(f"Generated dataset with {len(X)} samples")
    print(f"Legitimate samples: {sum(y == 0)}")
    print(f"Phishing samples: {sum(y == 1)}")
