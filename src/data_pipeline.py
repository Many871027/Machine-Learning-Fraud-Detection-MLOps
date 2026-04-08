import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(filepath="creditcard.csv", save_encoders=False):
    """
    Loads Credit Card Fraud CSV. No major preprocessing needed as data is purely numerical (PCA).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Credit Card dataset does not natively contain missing values, but just in case:
    df = df.dropna()
    
    # Return empty string dict to keep API signature consistent with main backend logic
    encoders = {}
    return df, encoders

def preprocess_new_data(new_data_dict, encoders_path=None):
    """
    Preprocess new inference data directly into a DataFrame.
    """
    df = pd.DataFrame([new_data_dict])
    return df
