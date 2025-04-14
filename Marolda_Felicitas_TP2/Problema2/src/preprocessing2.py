import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import models2 as mod2
import metrics2 as met2

def prepare_df(df):
    #sacamos los valores nulos
    df = df.dropna()
    return df


def normalization(X_, mu = None, sigma = None):
    X = X_.copy()
    if mu is None:
        mu = median(X)
    if sigma is None:
        sigma = std(X)

    X = (X - mu) / sigma
    return X

def median(X):
    return np.median(X, axis=0)

def std(X):
    return np.std(X, axis=0)

def feature_engineering(df):
    df['mp_x_poss'] = df['mp']*df['poss']
    df['poss_x_raptor_total'] = df['poss']*df['raptor_total']
    df['raptor_total_x_war_total'] = df['raptor_total']*df['war_total']
    df['war_total_x_pace_impact'] = df['war_total']*df['pace_impact']
    
    return df


def split_data(X: pd.DataFrame, validation_size: float = 0.2) -> tuple:
    """X: data original
    validation_size: tamaño del conjunto de validación
    Devuelve X_train, X_val"""
    X_ = X.copy()
    X_val = X_.sample(frac=validation_size, random_state=42)
    X_train = X_.drop(X_val.index)
    return X_train, X_val

def df_breakDown(df_, target_column='y'):
    df = df_.copy()
    """Descompone un DataFrame en X, y y las features"""
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    features = df.drop(columns=[target_column]).columns  

    y = y.reshape(-1, 1)
    return X, y, features


def cross_validation_for_LogisticReg(df_dev_, possible_L2, folds: int = 10):
    df_dev = df_dev_.copy()
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

            X_train, y_train, features = df_breakDown(X_train_fold, 'war_class')
            X_val, y_val, _ = df_breakDown(X_val_fold, 'war_class')

            # undersampling
            X_train, y_train = undersampling(X_train, y_train)

            # Normalización con media y std del training
            X_train = normalization(X_train)
            X_val = normalization(X_val, median(X_train), std(X_train))

            # Entrenar y predecir
            model = mod2.Logistic_Regression_Multiclass(X_train, y_train, features, L2=L2, threshold=0.5)
            predictions = model.predict(X_val)

            # Calcular f-score
            fscore = met2.f_score_multiclass(y_val, predictions)
            print("Fscore:", fscore)
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
    print("poss l2: ", possible_L2)
    print("fscore: ", fscore_path)
    met2.graph_val_fscore(possible_L2, fscore_path)
    return best_L2

def undersampling(X, y):
    X_balanced = X.copy()
    y_balanced = y.copy()
    
    # Count occurrences of each class
    unique_classes, class_counts = np.unique(y_balanced, return_counts=True)
    min_count = np.min(class_counts)  # Find the minority class count

    # Create a mask to keep track of indices to retain
    indices_to_keep = np.array([], dtype=int)

    for cls in unique_classes:
        # Get indices of the current class
        class_indices = np.where(y_balanced == cls)[0]
        # Randomly select min_count indices from the current class
        selected_indices = np.random.choice(class_indices, min_count, replace=False)
        # Append selected indices to the mask
        indices_to_keep = np.concatenate((indices_to_keep, selected_indices))

    # Subset X and y using the indices to keep
    X_balanced = X_balanced[indices_to_keep]
    y_balanced = y_balanced[indices_to_keep]

    return X_balanced, y_balanced

def cross_validation_for_RanfomForest(df_dev_, type, possible_hyp, params, folds = 10):
    df_dev = df_dev_.copy()
    best_fscore = -1
    best_hyp = None
    fscore_path = []
    df_dev = df_dev.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(df_dev) // folds
    for hyp in possible_hyp:
        print(f"Testing {type}={hyp}")
        fscores = []
        for fold in range(folds):
            # print(f"Testing {type} with fold {fold}")
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != folds - 1 else len(df_dev)
            X_val_fold = df_dev.iloc[start:end]
            X_train_fold = pd.concat([df_dev.iloc[:start], df_dev.iloc[end:]])

            X_train, y_train, features = df_breakDown(X_train_fold, 'war_class')
            X_val, y_val, _ = df_breakDown(X_val_fold, 'war_class')

            # undersampling
            X_train, y_train = undersampling(X_train, y_train)

            # Normalización con media y std del training
            X_train = normalization(X_train)
            X_val = normalization(X_val, median(X_train), std(X_train))

            # Entrenar y predecir
            if type == "n_trees":
                # print(f"Testing n_trees={hyp}")
                model = mod2.RandomForest(X_train, y_train, features, hyp, params['max_depth'], params['perc_features'])
            elif type == "max_depth":
                # print(f"Testing max_depth={hyp}")
                model = mod2.RandomForest(X_train, y_train, features, params['n_trees'], hyp, params['perc_features'])
            elif type == "perc_features":
                # print(f"Testing features_percent={hyp}")
                model = mod2.RandomForest(X_train, y_train, features, params['n_trees'], params['max_depth'], hyp)
            else:

                raise ValueError("Invalid type. Choose from 'n_trees', 'max_depth', or 'features_percent'.")

            predictions = model.predict(X_val)

            # Calcular f-score
            fscore = met2.f_score_multiclass(y_val, predictions)
            fscores.append(fscore)
            print("Fscore:", fscore)

        avg_fscore = np.mean(fscores)
        fscore_path.append(avg_fscore)
        if avg_fscore > best_fscore:
            best_fscore = avg_fscore
            best_hyp = hyp
    
    print("hyp: ", possible_hyp)
    print("fscore: ", fscore_path)
    met2.graph_val_fscore(possible_hyp, fscore_path)
    return best_hyp

