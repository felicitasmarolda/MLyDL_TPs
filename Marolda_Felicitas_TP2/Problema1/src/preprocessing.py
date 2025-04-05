import numpy as np
import pandas as pd
import models as mod

def prepare_df(df_):
    """Recibimos el dataframe original y devolvemos uno ya cambiado y procesado.
    Cambios:
        - hacemos binaria la columna de GeneticMutation
        - con KNN obtenemos los ??? de CellType y los agregamos al df
        - hacemos one hot encoding de CellType
        - eliminamos GeneticMutation y CellType
        - los nan los llenamos con la media de cada columna (si la columna es binaria la dropeamos)
     """
    df = df_.copy()

    df['GeneticMutationBinary'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    df = df.drop(columns=['GeneticMutation'], errors='ignore')

    df['CellTypeEncoded'], uniques = pd.factorize(df['CellType'])
    print(uniques)
    df = df.drop(columns=['CellType'], errors='ignore')

    # convertimos a array y dividimos en X, y y las features
    # guardamos la columna 'Diagnosis' en una lista
    diagnosis = df['Diagnosis'].values
    df = df.drop(columns=['Diagnosis'], errors='ignore')    # ahora el dataframe no tiene la columna de diagnosis

    df = df.dropna(subset=["CellTypeEncoded"])
    columnas_a_rellenar = df.columns.difference(["CellTypeEncoded"])
    df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(df[columnas_a_rellenar].mean())

    features_names = list(df.columns)
    print("Features names:", features_names)

    # ahora ya no hay nans
    X, y, features = df_breakDown(df, y='CellTypeEncoded')
    y = np.array(y).ravel()
    X_with = []
    cellType_with = []
    X_without = []
    cellType_without = []
    diagnosis_with = []
    diagnosis_without = []      # agregamos los diagnosis para cuando hagamos el concatenate no se pierda el orde

    for i in range(len(y)):
        if y[i] == 1:
            cellType_with.append(y[i])
            X_with.append(X[i])
            diagnosis_with.append(diagnosis[i])
        else:
            cellType_without.append(y[i])
            X_without.append(X[i])
            diagnosis_without.append(diagnosis[i])
    
    model_knn = mod.KNN(X_with, cellType_with, features, k=3)
    predictions_cell_type = model_knn.predict(X_without)

    # reunimos todo
    X = np.concatenate((X_with, X_without), axis=0)
    cellType = np.concatenate((cellType_with, predictions_cell_type), axis=0)
    diagnosis = np.concatenate((diagnosis_with, diagnosis_without), axis=0)
    
    new_df = pd.DataFrame(X, columns=features)
    new_df['CellTypeEncoded'] = cellType
    new_df['Diagnosis'] = diagnosis

    return new_df






    

def prepare_for_knn_cell_type(df):
    """Descompone un DataFrame en X, y y las features y saca la columna de prediccion"""
    df = df.copy()

    # df = df.dropna() 
    cols_to_dropna = ['CellType']
    df = df.dropna(subset=cols_to_dropna)

    df['GeneticMutationBinary'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    df['CellTypeEncoded'], uniques = pd.factorize(df['CellType'])
    print(uniques)
    # convertimos a array y dividimos en X, y y las features

    # guardamos la columna 'Diagnosis' en una lista
    diagnosis = df['Diagnosis'].values

    df = df.drop(columns=['CellType', 'GeneticMutation', 'Diagnosis'], errors='ignore')
    cols_to_fill_mean = df.columns.difference(['A', 'B'])
    df[cols_to_fill_mean] = df[cols_to_fill_mean].fillna(df[cols_to_fill_mean].mean())  
    features_names = list(df.columns)


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
        if y[i] == 1:
            y_predict.append(y[i])
            X_predict.append(X[i])
        else:
            y_train.append(y[i])
            X_train.append(X[i])

    return X_train, y_train, features, X_predict, y_predict, diagnosis, features_names

# ESTA MAL EL AGREGADO DE DIAGNOSIS

def reunite_cell_type(X_prev, y_prev, X_new, y_new, diagnosis, features_names):
    """Reune los datos pero ahora con cell type decodificado, le vuelve a sumar la 
    columna de diagnósticos.
    """
    X = np.concatenate((X_prev, X_new), axis=0)
    y = np.concatenate((y_prev, y_new), axis=0)

    new_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    new_df['CellTypeEncoded'] = y
    new_df['Diagnosis'] = diagnosis

    # le ponemos nombres a las columnas
    new_df.columns = features_names + ['Diagnosis']


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

