import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
folder_path = r'C:\Users\harsh\Desktop\ieee project\data'

def load_and_merge():
    print("Loading datasets...")
    
    # 1. Load separately to manage memory
    train_trans = pd.read_csv(os.path.join(folder_path, 'train_transaction.csv'))
    train_id = pd.read_csv(os.path.join(folder_path, 'train_identity.csv'))
    
    # 2. Merge and then immediately delete components to free RAM
    print("Merging training data...")
    train = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    del train_trans, train_id
    gc.collect() # Force garbage collection
    
    # 3. Repeat for test
    test_trans = pd.read_csv(os.path.join(folder_path, 'test_transaction.csv'))
    test_id = pd.read_csv(os.path.join(folder_path, 'test_identity.csv'))
    
    print("Merging test data...")
    test = pd.merge(test_trans, test_id, on='TransactionID', how='left')
    del test_trans, test_id
    gc.collect()
    
    return train, test

# Execute
train, test = load_and_merge()

print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
