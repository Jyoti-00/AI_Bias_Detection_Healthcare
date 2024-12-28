import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(data)} rows.")
    return data

def preprocess_data(data, sample_size=100):
    """Preprocess the dataset: sampling, splitting, and formatting."""
    # Sample the dataset for computational efficiency
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from the dataset.")
    
    # Select relevant columns
    columns_to_keep = ['sentence_text', 'stereotype', 'predicted gender']
    data = data[columns_to_keep]
    
    # Split into training and testing datasets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_data)} rows, Test set: {len(test_data)} rows.")

    # Convert to Hugging Face dataset format
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    return train_dataset, test_dataset

if __name__ == "__main__":
    file_path = "data/balanced_BUG.csv"  # Update the path as necessary
    data = load_data(file_path)
    train_dataset, test_dataset = preprocess_data(data)
