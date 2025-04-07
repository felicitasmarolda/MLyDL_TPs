import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models as mod
import metrics as met

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
    # diagnosis = df['Diagnosis'].values
    # df = df.drop(columns=['Diagnosis'], errors='ignore')    # ahora el dataframe no tiene la columna de diagnosis

    df = df.dropna(subset=["CellTypeEncoded"])
    columnas_a_rellenar = df.columns.difference(["CellTypeEncoded"])
    df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(df[columnas_a_rellenar].mean())

    # features_names = list(df.columns)
    # print("Features names:", features_names)

    # # ahora ya no hay nans
    # X, y, features = df_breakDown(df, y='CellTypeEncoded')
    # y = np.array(y).ravel()
    # X_with = []
    # cellType_with = []
    # X_without = []
    # cellType_without = []
    # diagnosis_with = []
    # diagnosis_without = []      # agregamos los diagnosis para cuando hagamos el concatenate no se pierda el orde

    # for i in range(len(y)):
    #     if y[i] == 1:
    #         cellType_with.append(y[i])
    #         X_with.append(X[i])
    #         diagnosis_with.append(diagnosis[i])
    #     else:
    #         cellType_without.append(y[i])
    #         X_without.append(X[i])
    #         diagnosis_without.append(diagnosis[i])
    
    # model_knn = mod.KNNClassifier(X_with, cellType_with, features, k=3)
    # predictions_cell_type = model_knn.predict(X_without)

    # # reunimos todo
    # X = np.concatenate((X_with, X_without), axis=0)
    # cellType = np.concatenate((cellType_with, predictions_cell_type), axis=0)
    # diagnosis = np.concatenate((diagnosis_with, diagnosis_without), axis=0)
    
    # new_df = pd.DataFrame(X, columns=features)
    # new_df['CellTypeEncoded'] = cellType
    # new_df['Diagnosis'] = diagnosis
    print(df.describe())
    return df
    return new_df

def knn_for_nans(X, k = 4):
    """Recibimos un X df y devolvemos el mismo df pero donde hay nan hacemos knn y 
    ponemos el promedio."""
    # va a haber que dividir los datos devuela en 2, despues vemos eso

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

def normalization(X: pd.DataFrame, mu, sigma) -> pd.DataFrame:
    """X: data original
    mu: media de la columna
    sigma: desviación estándar de la columna
    Devuelve X normalizado"""
    X = (X - mu) / (sigma - mu)
    return X


def cross_validation_for_L2(df_dev, possible_L2, folds: int = 5, validation_size = 0.2):
    """X: data original
    y: labels
    folds: cantidad de folds para cross validation
    possible_L2: lista de posibles valores de L2 para probar
    thresholds: lista de thresholds para probar
    Prueba diferentes valores de L2 y del threshold para encontrar el óptimo de L2 usando fscore"""

    best_fscore = -1
    best_L2 = None
    fscore_path = []

    df_dev = df_dev.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_dev) // folds

    for L2 in possible_L2:
        # print(f"Testing L2={L2}")
        fscores = []
        for fold in range(folds):

            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != folds - 1 else len(df_dev)
            X_val_fold = df_dev.iloc[start:end]
            X_train_fold = pd.concat([df_dev.iloc[:start], df_dev.iloc[end:]])

            X_train, y_train, features = df_breakDown(X_train_fold, y='Diagnosis')
            X_val, y_val, _ = df_breakDown(X_val_fold, y='Diagnosis')

            # Normalización con media y std del training
            X_train = normalization(X_train, X_train.mean(), X_train.std())
            X_val = normalization(X_val, X_train.mean(), X_train.std())

            # Entrenar y predecir
            model = mod.Logistic_Regression(X_train, y_train, features, L2=L2, threshold=0.5)
            predictions = model.predict(X_val)

            # Calcular f-score
            fscore = met.f_score(y_val, predictions)
            # print("Fscore:", fscore)
            fscores.append(fscore)

        # Esto va fuera del loop de folds
        avg_fscore = np.mean(fscores)
        fscore_path.append(avg_fscore)

        # print(f"Avg fscore for L2={L2}: {avg_fscore}")

        if avg_fscore > best_fscore:
            # print(f"New best fscore: {avg_fscore} for L2={L2}")
            best_fscore = avg_fscore
            best_L2 = L2

    # Graficar resultados
    met.graph_L2_fscore(possible_L2, fscore_path)
    return best_L2

def cross_validation_for_threshold(df_dev, L2, thresholds: list, folds: int = 5, validation_size = 0.2):
    """X: data original
    y: labels
    folds: cantidad de folds para cross validation
    thresholds: lista de thresholds para probar
    Prueba diferentes valores de L2 y del threshold para encontrar el óptimo de L2 usando fscore"""

    best_fscore = -1
    best_threshold = None
    fscore_path = []

    df_dev = df_dev.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_dev) // folds

    for threshold in thresholds:
        # print(f"Testing threshold={threshold}")
        fscores = []
        for fold in range(folds):

            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != folds - 1 else len(df_dev)
            X_val_fold = df_dev.iloc[start:end]
            X_train_fold = pd.concat([df_dev.iloc[:start], df_dev.iloc[end:]])

            X_train, y_train, features = df_breakDown(X_train_fold, y='Diagnosis')
            X_val, y_val, _ = df_breakDown(X_val_fold, y='Diagnosis')

            # Normalización con media y std del training
            X_train = normalization(X_train, X_train.mean(), X_train.std())
            X_val = normalization(X_val, X_train.mean(), X_train.std())

            # Entrenar y predecir
            model = mod.Logistic_Regression(X_train, y_train, features, L2=L2, threshold=threshold)
            predictions = model.predict(X_val)

            # Calcular f-score
            fscore = met.f_score(y_val, predictions)
            # print("Fscore:", fscore)
            fscores.append(fscore)

        # Esto va fuera del loop de folds
        avg_fscore = np.mean(fscores)
        fscore_path.append(avg_fscore)

        # print(f"Avg fscore for threshold={threshold}: {avg_fscore}")

        if avg_fscore > best_fscore:
            # print(f"New best fscore: {avg_fscore} for threshold={threshold}")
            best_fscore = avg_fscore
            best_threshold = threshold

    # Graficar resultados
    met.graph_L2_fscore(thresholds, fscore_path)
    return best_threshold