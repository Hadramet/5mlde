import pandas as pd
from preprocessing import preprocess_data

# Read the sample dataset
sample_data = pd.read_csv("sample_data.csv")

# Print the original data
print("Original data:")
print(sample_data)

# Preprocess the data
preprocessed_data = preprocess_data(sample_data)

# Print the preprocessed data
print("\nPreprocessed data:")
print(preprocessed_data)