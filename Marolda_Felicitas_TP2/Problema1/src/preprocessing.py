import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models as mod
import metrics as met

def prepare_df(df_):
    """Recibimos el dataframe original y devolvemos uno ya cambiado y procesado.
    Cambios:
        - hacemos binaria la columna de GeneticMutation
        - hacemos one hot encoding de CellType
        - eliminamos GeneticMutation y CellType
        - los nan los llenamos con la mediana de cada columna (si la columna es binaria la dropeamos)
        - hacemos que los datos esten el rango especificado en su descripción
     """
    df = df_.copy()

    df['GeneticMutationBinary'] = (df['GeneticMutation'] == 'Presnt').astype(int)
    df = df.drop(columns=['GeneticMutation'], errors='ignore')

    encoded = pd.get_dummies(df['CellType'], prefix='CellType')
    encoded.columns = ['Unknown', 'Epthlial', 'Mesnchymal']  
    df = pd.concat([df.drop(columns=['CellType'], errors='ignore'), encoded], axis=1)

    df = df.dropna(subset=["GeneticMutationBinary", "Unknown", "Epthlial", "Mesnchymal"])
    columnas_a_rellenar = df.columns.difference(["GeneticMutationBinary","Unknown", "Epthlial", "Mesnchymal"])
    df[columnas_a_rellenar] = df[columnas_a_rellenar].fillna(df[columnas_a_rellenar].median())
    df['Unknown'] = df['Unknown'].astype(int)
    df['Epthlial'] = df['Epthlial'].astype(int)
    df['Mesnchymal'] = df['Mesnchymal'].astype(int)
    
    df = fit_df_to_prespecified_bounds(df)

    print(df.head(1))
    return df
    return new_df

def prepare_df_test(df_test_, df_dev_):
    df_test_ = df_test_.copy()
    df_dev_ = df_dev_.copy()

    df_test_['GeneticMutationBinary'] = (df_test_['GeneticMutation'] == 'Presnt').astype(int)
    df_test_ = df_test_.drop(columns=['GeneticMutation'], errors='ignore')

    encoded = pd.get_dummies(df_test_['CellType'], prefix='CellType')
    encoded.columns = ['Unknown', 'Epthlial', 'Mesnchymal']  
    df_test_ = pd.concat([df_test_.drop(columns=['CellType'], errors='ignore'), encoded], axis=1) 
    df_test_['Unknown'] = df_test_['Unknown'].astype(int)
    df_test_['Epthlial'] = df_test_['Epthlial'].astype(int)
    df_test_['Mesnchymal'] = df_test_['Mesnchymal'].astype(int)
    
    df_test_ = df_test_.dropna(subset=["GeneticMutationBinary", "Unknown", "Epthlial", "Mesnchymal"])
    columnas_a_rellenar = df_test_.columns.difference(["GeneticMutationBinary","Unknown", "Epthlial", "Mesnchymal"])
    df_test_[columnas_a_rellenar] = df_test_[columnas_a_rellenar].fillna(df_dev_[columnas_a_rellenar].median())

    df_test_ = fit_df_to_prespecified_bounds(df_test_)

    return df_test_

def fit_df_to_prespecified_bounds(df):
    """Elimina los outliers de mi df"""
    df.loc[df['CellAdhesion'] > 1, 'CellAdhesion'] = 1
    df.loc[df['CellAdhesion'] < 0, 'CellAdhesion'] = 0

    df.loc[df['NuclearMembrane'] > 5, 'NuclearMembrane'] = 5
    df.loc[df['NuclearMembrane'] < 1, 'NuclearMembrane'] = 1

    df.loc[df['Vascularization'] < 0, 'Vascularization'] = 0
    df.loc[df['Vascularization'] > 10, 'Vascularization'] = 10

    df.loc[df['InflammationMarkers'] < 0, 'InflammationMarkers'] = 0
    df.loc[df['InflammationMarkers'] > 100, 'InflammationMarkers'] = 100

    return df

def remove_outliers(X, bounds:dict):
    for column in range(X.shape[1]):
        lower_bound, upper_bound = bounds[column]
        X[:, column] = np.clip(X[:, column], lower_bound, upper_bound)
    
    return X
    

def get_bounds(X, std_multiplier=3):
    bounds = {}
    for column in range(X.shape[1]):
        median = np.median(X[:, column])
        std = np.std(X[:, column])
        lower_bound = median - std_multiplier * std
        upper_bound = median + std_multiplier * std
        bounds[column] = (lower_bound, upper_bound)
    return bounds

def df_breakDown(df, y='y'):
    target_column = y
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    features = df.drop(columns=[target_column]).columns  

    y = y.reshape(-1, 1)
    return X, y, features

def split_data(X: pd.DataFrame, validation_size: float = 0.2) -> tuple:
    X_ = X.copy()
    X_val = X_.sample(frac=validation_size, random_state=42)
    X_train = X_.drop(X_val.index)
    return X_train, X_val

def normalization(X, mu = None, sigma = None, bounds = None):
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0)
    if bounds is None:
        bounds = get_bounds(X)

    X = remove_outliers(X, bounds)

    X = (X - mu) / sigma
    return X

def median(X):
    return np.median(X, axis=0)

def std(X):
    return np.std(X, axis=0)



def cross_validation_for_L2(df_dev, possible_L2, folds: int = 10, validation_size = 0.2):
    best_fscore = -1
    best_L2 = None
    fscore_path = []

    df_dev = df_dev.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_dev) // folds

    for L2 in possible_L2:
        fscores = []
        for fold in range(folds):

            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != folds - 1 else len(df_dev)
            X_val_fold = df_dev.iloc[start:end]
            X_train_fold = pd.concat([df_dev.iloc[:start], df_dev.iloc[end:]])

            X_train, y_train, features = df_breakDown(X_train_fold, y='Diagnosis')
            X_val, y_val, _ = df_breakDown(X_val_fold, y='Diagnosis')

            # print("Type: ", type(X_train), type(X_val))
            # print("shape: ", X_train.shape, X_val.shape)
            # print("mean: ", np.mean(X_train, axis = 0))
            X_train = normalization(X_train)
            X_val = normalization(X_val, median(X_train), std(X_train), get_bounds(X_train))

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

def cross_validation_for_threshold(df_dev, L2, thresholds: list, folds: int = 10, validation_size = 0.2):

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

            X_train = normalization(X_train)
            X_val = normalization(X_val, median(X_train), std(X_train), get_bounds(X_train))

            model = mod.Logistic_Regression(X_train, y_train, features, L2=L2, threshold=threshold)
            predictions = model.predict(X_val)

            fscore = met.f_score(y_val, predictions)
            # print("Fscore:", fscore)
            fscores.append(fscore)

        avg_fscore = np.mean(fscores)
        fscore_path.append(avg_fscore)

        # print(f"Avg fscore for threshold={threshold}: {avg_fscore}")

        if avg_fscore > best_fscore:
            # print(f"New best fscore: {avg_fscore} for threshold={threshold}")
            best_fscore = avg_fscore
            best_threshold = threshold

    met.graph_val_fscore(thresholds, fscore_path)
    return best_threshold

def cross_validation_for_imbalanced(df_dev, possible_L2, rebalanceo = None, folds: int = 10, validation_size = 0.2):
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
            
            X_train = normalization(X_train)
            X_val = normalization(X_val, median(X_train), std(X_train), get_bounds(X_train))

            model = mod.Logistic_Regression(X_train, y_train, features, L2=L2, threshold=0.5)
            predictions = model.predict(X_val)

            fscore = met.f_score(y_val, predictions)
            # print("Fscore:", fscore)
            fscores.append(fscore)

        avg_fscore = np.mean(fscores)
        fscore_path.append(avg_fscore)

        # print(f"Avg fscore for L2={L2}: {avg_fscore}")

        if avg_fscore > best_fscore:
            # print(f"New best fscore: {avg_fscore} for L2={L2}")
            best_fscore = avg_fscore
            best_L2 = L2

    met.graph_val_fscore(possible_L2, fscore_path)
    return best_L2

def undersampling(X, y):
    X_balanced = X.copy()
    y_balanced = y.copy()
    minority = 0
    mayority = 0
    for i in y_balanced:
        if i == 1:
            minority += 1
        else:
            mayority += 1
    
    while mayority > minority:
        index = np.random.randint(0, len(X_balanced))
        if y_balanced[index] == 0:
            X_balanced = np.delete(X_balanced, index, axis=0)
            y_balanced = np.delete(y_balanced, index, axis=0)
            mayority -= 1
    
    return X_balanced, y_balanced

def oversampling_duplication(X, y):
    X_balanced = X.copy()
    y_balanced = y.copy()
    minority = 0
    mayority = 0
    for i in y_balanced:
        if i == 1:
            minority += 1
        else:
            mayority += 1
    
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
        index = np.random.randint(0, len(X_balanced))
        if y_balanced[index] == 1:
            distances = np.linalg.norm(X_balanced - X_balanced[index], axis=1)
            neighbors = np.argsort(distances)[1:4]
            neighbor_index = np.random.choice(neighbors)
            lambda_ = np.random.rand()
            new_sample = X_balanced[index] + lambda_ * (X_balanced[neighbor_index] - X_balanced[index])
            X_balanced = np.append(X_balanced, [new_sample], axis=0)
            y_balanced = np.append(y_balanced, [y_balanced[index]])
            minority += 1
    
    return X_balanced, y_balanced

def cost_reweighting(X, y):

    X_ = X.copy()
    y_ = y.copy()
    counts = y.value_counts()
    majority_class = counts.idxmax()
    minority_count = counts.min()
    X_minority = X_[y_ != majority_class]
    y_minority = y_[y_ != majority_class]
    
    p1 = len(y_minority) / len(y)
    p2 = len(X_) / len(y)
    
    C = p2 / p1
    
    X_minority_weighted = X_minority * C
    
    X_balanced = pd.concat([X_[y_ == majority_class], X_minority_weighted])
    y_balanced = pd.concat([y_[y_ == majority_class], y_[y_ != majority_class]])
    
    return X_balanced, y_balanced