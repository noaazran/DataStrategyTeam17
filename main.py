import os
import numpy as np
import pandas as pd
import datetime as dt

from src.full_df_preprocessing import *

datatypes = {
    'product_id': 'uint32',
    'client_id': 'uint32',
    'sales_net': 'float64',
    'quantity': 'uint8',
    'order_channel': 'object',
    'branch_id': 'uint16'
}

def main():
    print("Loading raw dataset...")
    df = load_dataset(datatypes)
    
    print("Date columns preprocessing...")
    df = dates_preprocessing(df)
    
    print("Preprocessing order_channel column...")
    df = order_channel_preprocessing(df)

    print("Adding features...")
    df = add_features(df)
    
    print("Saving preprocessed dataset...")
    df.to_csv(os.path.join("data", "preprocessed_data.csv"))
    print("Done")
    return


if __name__=="__main__":
    main()