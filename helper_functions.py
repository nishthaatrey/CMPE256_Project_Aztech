import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
from sklearn import preprocessing

#Dummies the dataframe
def dummies(df, columns, special_col):
        
    for sub in columns:
        df[sub] = df[sub].apply(lambda x: str(x).replace(" ", "").split(";"))
        if sub == special_col:
            df[sub] = df[sub].apply(lambda x: ["Want_" + s for s in x])
        df = pd.concat([df, pd.get_dummies(pd.DataFrame(df[sub].tolist(), index=df.index).stack()).sum(level=0)], axis=1).drop(sub, axis=1)
    df = pd.get_dummies(df)
    return df

#Preprocess dataframe (dummies and nan)
def preprocessed(df, columns, special_col, prof):
    df = df.dropna()
    final_df = dummies(df.copy(), columns, special_col)
    if prof:
        final_df.JobSatisfaction /= 10
        final_df.CareerSatisfaction /= 10
    return final_df, df

#Compute knn graph using sklearn
def compute_knn_graph(df):
    graph = kneighbors_graph(df, int(np.sqrt(df.shape[0])), mode='distance', include_self=True)
    graph.data = np.exp(- graph.data ** 2 / (2. * np.mean(graph.data) ** 2))
    return graph