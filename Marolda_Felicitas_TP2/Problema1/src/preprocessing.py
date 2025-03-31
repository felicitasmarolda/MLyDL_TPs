import numpy as np
import pandas as pd

def fix_X(X: pd.DataFrame) -> pd.DataFrame:
    """X: data original
    Devuelve X con las modificaciones necesarias:
    - 
    """

    return X

def df_breakDown(df, y='y'):
    """Descompone un DataFrame en X, y y las features"""
    target_column = y
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    features = df.drop(columns=[target_column]).columns  

    y = y.reshape(-1, 1)
    return X, y, features
