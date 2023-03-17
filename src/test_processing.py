import pandas as pd
from training import  preprocess_data, train_model, batch_inference

path = "data/spam_emails_1.csv"

# test preprocess data
# result = preprocess_data(path)

# print result
# print(result)

# test model training and evaluation
train_model(path)

# test batch inference
result = batch_inference(path)
print(result)