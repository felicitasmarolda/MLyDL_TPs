import numpy as np
import pandas as pd

def prepare_for_knn_cell_type(df):
    """Descompone un DataFrame en X, y y las features y saca la columna de prediccion"""
    df = df.copy()

    df = df.dropna()

    df['GeneticMutationBinary'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    df['CellTypeEncoded'], uniques = pd.factorize(df['CellType'])
    print(uniques)
    # convertimos a array y dividimos en X, y y las features

    df = df.drop(columns=['CellType', 'GeneticMutation', 'Diagnosis'], errors='ignore')

    X, y, features = df_breakDown(df, y='CellTypeEncoded')
    y = np.array(y).ravel()
    # X = np.array(X).ravel()
    # print("y:", y)
    # print("X:", X)
    # recorremos el y y dividimos en los que tienen signos de pregunta y los que no como y_train, y_predict, lo mismo para X en base a lo de y
    y_train = []
    y_predict = []
    X_train = []
    X_predict = []
    for i in range(len(y)):
        if y[i] == 2:
            y_predict.append(y[i])
            X_predict.append(X[i])
        else:
            y_train.append(y[i])
            X_train.append(X[i])

    return X_train, y_train, features, X_predict, y_predict

def reunite_cell_type(X_prev, y_prev, X_new, y_new, df):
    """Reune los datos pero ahora con cell type decodificado, le vuelve a sumar la 
    columna de diagnósticos.
    """
    df = df.copy()
    # print("X_prev", X_prev)
    # print("y_prev", y_prev)
    # print("X_new", X_new)
    # print("y_new", y_new)
    # print("df", df)

    # unimos los datos
    X = np.concatenate((X_prev, X_new), axis=0)
    y = np.concatenate((y_prev, y_new), axis=0)

    diagnosis = df['Diagnosis'].values

    new_df = pd.DataFrame(X, columns=df.columns[:-1])
    new_df['Diagnosis'] = diagnosis
    new_df['CellTypeBinary'] = y

    print("new_df", new_df)

    return new_df


    
def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """df: data original
    Devuelve X con las modificaciones necesarias:
    - 
    """
    df = df.copy()
    # sacamos los nan
    df = df.dropna()

    # one hot encoding para cell type
    df['Epthlial'] = (df['CellType'] == 'Epthlial').astype(int)
    df['Mesnchymal'] = (df['CellType'] == 'Mesnchymal').astype(int)
    df['???'] = (df['CellType'] == '???').astype(int)   # DESPUES VER QU EHACER CON ESTO

    df['GeneticMutationBinary'] = (df['GeneticMutation'] == 'Presnt').astype(int)

    # eliminamos las columnas que no nos interesan
    df = df.drop(columns=['CellType', 'GeneticMutation'], errors='ignore')

    return df    




    return df

def df_breakDown(df, y='y'):
    """Descompone un DataFrame en X, y y las features"""
    target_column = y
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    features = df.drop(columns=[target_column]).columns  

    y = y.reshape(-1, 1)
    return X, y, features

def split_data(X: pd.DataFrame, validation_size: float = 0.2) -> tuple:
    """X: data original
    validation_size: tamaño del conjunto de validación
    Devuelve X_train, X_val"""
    X_ = X.copy()
    X_val = X_.sample(frac=validation_size, random_state=42)
    X_train = X_.drop(X_val.index)
    return X_train, X_val

def min_max_scaling(X: pd.DataFrame, min, max) -> pd.DataFrame:
    """X: data original
    Devuelve X escalado entre 0 y 1 por columna"""
    X_scaled = (X - min) / (max - min)
    return X_scaled

