import pandas as pd
import datetime as dt

def load_dataset(datatypes):
    return pd.read_csv('data/transactions_dataset.csv', dtype=datatypes, sep=';')

def dates_preprocessing(df): 
    df['date_order'] = pd.to_datetime(df['date_order'], format='%Y-%m-%d')
    df['date_invoice'] = pd.to_datetime(df['date_invoice'], format='%Y-%m-%d')
    return df

def order_channel_preprocessing(df):
    df.order_channel = df.order_channel.astype('category')
    return df

def add_features(df):
    # Add unit sale price
    df["unit_price"] = df["sales_net"] / df["quantity"]
    df.unit_price = df.unit_price.astype('int8')
    
    # Add flow : 1 means it is a purchase (increased stock), 0 it is a sale
    purchases = df['sales_net'] >= 0
    df["stock_flow"] = 0
    df.loc[purchases, 'stock_flow'] = 1
    df.stock_flow = df.stock_flow.astype('int8')
    
    df["quantity"] = df["quantity"] * df["stock_flow"]
    df.quantity = df.quantity.astype('int32')
    
    # Date features
    df["month_order"] = df["date_order"].dt.month
    df["order_invoice_delta"] = (df["date_invoice"] - df["date_order"]).dt.days
    return df