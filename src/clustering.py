import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from kneed import KneeLocator

def load_preprocessed_data(datatypes):
    df = pd.read_csv('data/preprocessed_data.csv', dtype=datatypes, sep=',')
    for col in ['Unnamed: 0', 'date_order', 'date_invoice']:
        if col in df.columns:
            df.drop(columns=col, axis=1, inplace=True)
    return df


def categorical_encoding(df):
    le = LabelEncoder()
    df.order_channel = le.fit_transform(df.order_channel)
    return df

def drop_na(df):
    return df.dropna(inplace=True)


def kmeans_clustering(df, min_n=2, max_n=8):
    """
    Description:
    creates clusters using k-means
    --------------------------------------------
    Args:
        df: input dataframe
        min_n: max number of clusters in the range(min, max) of KneeLocator
        max_n: max number of clusters in the range(min, max) of KneeLocator

    Returns: a dataframe with clusters
    """
    kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 750,
            "random_state": 0
        }
    # A list holds the SSE values for each k
    sse = []
    for k in range(min_n, max_n):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(
        range(min_n, max_n), sse, curve="convex", direction="decreasing"
    )

    kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs).fit(df)

    df_kmeans = pd.DataFrame(kmeans.transform(df))
    df_kmeans.columns = [str(i) for i in range(kl.elbow)]
    df_kmeans['cluster'] = kmeans.predict(df)
    return df_kmeans