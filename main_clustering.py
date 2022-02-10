import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from kneed import KneeLocator

from src.clustering import *

datatypes = {
    'product_id': 'uint32',
    'client_id': 'uint32',
    'sales_net': 'float64',
    'quantity': 'int32',
    'order_channel': 'object',
    'branch_id': 'uint16',
    'unit_price': 'int8',
    'stock_flow': 'int8',
    'month_order': 'uint8',
    'order_invoice_delta': 'float16'
}

def main():
    print("Loading preprocessed dataset...")
    df = load_preprocessed_data(datatypes)
    
    print("Encoding categorical column...")
    df = categorical_encoding(df)
    df.dropna(inplace=True)

    print("Clustering...")
    df_kmeans = kmeans_clustering(df, min_n=2, max_n=8)
    
    print("Saving cleaned data set...")
    df_kmeans.to_csv(os.path.join("data", "kmeans_data.csv"))
    print("Done")
    return


if __name__=="__main__":
    main()