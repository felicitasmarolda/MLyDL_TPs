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

def df_breakDown(df, target_column='y'):
    """Descompone un DataFrame en X, y y las features"""
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    features = df.drop(columns=[target_column]).columns  

    y = y.reshape(-1, 1)
    return X, y, features

def normalization(X: pd.DataFrame, mu, sigma) -> pd.DataFrame:
    """X: data original
    mu: media de la columna
    sigma: desviación estándar de la columna
    Devuelve X normalizado"""
    X = (X - mu) / (sigma - mu)
    return X

def cross_validation_for_LogisticReg(df_dev, possible_L2, folds: int = 5):
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

            X_train, y_train, features = df_breakDown(X_train_fold, 'war_class')
            X_val, y_val, _ = df_breakDown(X_val_fold, 'war_class')

            # Normalización con media y std del training
            X_train = normalization(X_train, X_train.mean(), X_train.std())
            X_val = normalization(X_val, X_train.mean(), X_train.std())

            # Entrenar y predecir
            model = mod2.Logistic_Regression_Multiclass(X_train, y_train, features, L2=L2, threshold=0.5)
            predictions = model.predict(X_val)

            # Calcular f-score
            fscore = met2.f_score_multiclass(y_val, predictions)
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
    met2.graph_val_fscore(possible_L2, fscore_path)
    return best_L2
