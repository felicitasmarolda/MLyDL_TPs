import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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