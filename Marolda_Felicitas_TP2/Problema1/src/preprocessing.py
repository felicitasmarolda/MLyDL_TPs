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

    # hacemos one hot encoding para CellType (una columna binaria para cad tipo)
    encoded = pd.get_dummies(df['CellType'], prefix='CellType')
    encoded.columns = ['Unknown', 'Epthlial', 'Mesnchymal']  # Asignás nombres fijos
    df = pd.concat([df.drop(columns=['CellType'], errors='ignore'), encoded], axis=1)
    # print("columns: ", df.columns)

    # convertimos a array y dividimos en X, y y las features
    # guardamos la columna 'Diagnosis' en una lista
    # diagnosis = df['Diagnosis'].values
    # df = df.drop(columns=['Diagnosis'], errors='ignore')    # ahora el dataframe no tiene la columna de diagnosis

    df = df.dropna(subset=["GeneticMutationBinary", "Unknown", "Epthlial", "Mesnchymal"])
    columnas_a_rellenar = df.columns.difference(["GeneticMutationBinary","Unknown", "Epthlial", "Mesnchymal"])
    df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(df[columnas_a_rellenar].median())
    # pasamos a 0 y 1 las de recien
    df['Unknown'] = df['Unknown'].astype(int)
    df['Epthlial'] = df['Epthlial'].astype(int)
    df['Mesnchymal'] = df['Mesnchymal'].astype(int)


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
    # print(df.describe())
    print(df.head(1))
    return df
    return new_df

def prepare_df_test(df_test_, df_dev_):
    df_test_ = df_test_.copy()
    df_dev_ = df_dev_.copy()

    df_test_['GeneticMutationBinary'] = (df_test_['GeneticMutation'] == 'Presnt').astype(int)
    df_test_ = df_test_.drop(columns=['GeneticMutation'], errors='ignore')

    encoded = pd.get_dummies(df_test_['CellType'], prefix='CellType')
    encoded.columns = ['Unknown', 'Epthlial', 'Mesnchymal']  # Asignás nombres fijos
    df_test_ = pd.concat([df_test_.drop(columns=['CellType'], errors='ignore'), encoded], axis=1)  # 🔥 Esta línea ya hace todo
    df_test_['Unknown'] = df_test_['Unknown'].astype(int)
    df_test_['Epthlial'] = df_test_['Epthlial'].astype(int)
    df_test_['Mesnchymal'] = df_test_['Mesnchymal'].astype(int)
    

    df_test_ = df_test_.dropna(subset=["GeneticMutationBinary", "Unknown", "Epthlial", "Mesnchymal"])
    columnas_a_rellenar = df_test_.columns.difference(["GeneticMutationBinary","Unknown", "Epthlial", "Mesnchymal"])
    df_test_[columnas_a_rellenar] = df_test_[columnas_a_rellenar].fillna(df_dev_[columnas_a_rellenar].median())

    return df_test_

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
    met.graph_val_fscore(possible_L2, fscore_path)
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
    met.graph_val_fscore(thresholds, fscore_path)
    return best_threshold

def cross_validation_for_imbalanced(df_dev, possible_L2, rebalanceo = None, folds: int = 5, validation_size = 0.2):
    """X: data original
    y: labels
    folds: cantidad de folds para cross validation
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

            # rebalanceo
            if rebalanceo == 'undersampling':
                X_train, y_train = undersampling(X_train, y_train)
                # print("Undersampling")
            elif rebalanceo == 'oversampling mediante SMOTE':
                X_train, y_train = oversampling_SMOTE(X_train, y_train)
                # print("Oversampling SMOTE")
            elif rebalanceo == 'oversampling mediante duplicación':
                X_train, y_train = oversampling_duplication(X_train, y_train)
                # print("Oversampling duplicación")
            elif rebalanceo == 'cost re-weighting':
                X_train, y_train = cost_reweighting(X_train, y_train)
                # print("Cost re-weighting")
            
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
    met.graph_val_fscore(possible_L2, fscore_path)
    return best_L2

def undersampling(X, y):
    """eliminar muestras de la clase mayoritaria de manera aleatoria
    hasta que ambas clases tengan igual proporción."""
    # Contamos la cantidad de datos de cada clasificacion
    X_balanced = X.copy()
    y_balanced = y.copy()
    minority = 0
    mayority = 0
    for i in y_balanced:
        if i == 1:
            minority += 1
        else:
            mayority += 1
    
    # Sacamos aleatoriamente datos de la clase mayoritaria
    while mayority > minority:
        index = np.random.randint(0, len(X_balanced))
        if y_balanced[index] == 0:
            X_balanced = np.delete(X_balanced, index, axis=0)
            y_balanced = np.delete(y_balanced, index, axis=0)
            mayority -= 1
    
    return X_balanced, y_balanced

def oversampling_duplication(X, y):
    """Oversampling mediante duplicación: duplicar muestras de la clase minoritaria
    de manera aleatoria, hasta que que ambas clases tengan igual proporción"""
    X_balanced = X.copy()
    y_balanced = y.copy()
    # Contamos la cantidad de datos de cada clasificacion
    minority = 0
    mayority = 0
    for i in y_balanced:
        if i == 1:
            minority += 1
        else:
            mayority += 1
    
    # Sacamos aleatoriamente datos de la clase mayoritaria
    while mayority > minority:
        index = np.random.randint(0, len(X_balanced))
        if y_balanced[index] == 1:
            X_balanced = np.append(X_balanced, [X_balanced[index]], axis=0)
            y_balanced = np.append(y_balanced, [y_balanced[index]])
            minority += 1
    
    return X_balanced, y_balanced

def oversampling_SMOTE(X, y):
    X_balanced = X.copy()
    y_balanced = y.copy()
    mayority = 0
    minority = 0
    for i in y_balanced:
        if i == 1:
            minority += 1
        else:
            mayority += 1
    
    while mayority > minority:
        # Elegimos aleatoriamente un dato de la clase minoritaria
        index = np.random.randint(0, len(X_balanced))
        if y_balanced[index] == 1:
            # Encontramos los KNN de ese dato
            distances = np.linalg.norm(X_balanced - X_balanced[index], axis=1)
            neighbors = np.argsort(distances)[1:4]
            # Elegimos un vecino aleatoriamente
            neighbor_index = np.random.choice(neighbors)
            # Generamos un nuevo dato entre el dato original y el vecino
            lambda_ = np.random.rand()
            new_sample = X_balanced[index] + lambda_ * (X_balanced[neighbor_index] - X_balanced[index])
            # Agregamos el nuevo dato al conjunto de datos
            X_balanced = np.append(X_balanced, [new_sample], axis=0)
            y_balanced = np.append(y_balanced, [y_balanced[index]])
            minority += 1
    
    return X_balanced, y_balanced

def cost_reweighting(X, y):
    # Contamos la cantidad de ejemplos de cada clase
    counts = y.value_counts()
    # Encontramos la clase mayoritaria
    majority_class = counts.idxmax()
    # Encontramos la cantidad de ejemplos de la clase minoritaria
    minority_count = counts.min()
    # Hacemos oversampling de la clase minoritaria
    X_minority = X[y != majority_class]
    y_minority = y[y != majority_class]
    
    # Calculamos las probabilidades a-priori
    p1 = len(y_minority) / len(y)
    p2 = len(X) / len(y)
    
    # Calculamos el factor de re-weighting
    C = p2 / p1
    
    # Re-weighting los datos de la clase minoritaria
    X_minority_weighted = X_minority * C
    
    # Concatenamos los datos ponderados con los de la clase mayoritaria
    X_balanced = pd.concat([X[y == majority_class], X_minority_weighted])
    y_balanced = pd.concat([y[y == majority_class], y[y != majority_class]])
    
    return X_balanced, y_balanced