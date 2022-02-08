import os
import pandas as pd
import datetime as dt


def load_raw_transactions():
    raw_transactions = pd.read_csv(os.path.join("data", "transactions_dataset.csv"), sep = ";")
    return raw_transactions


def select_transactions(raw_transactions):
    filtered_transactions = raw_transactions[raw_transactions.order_channel == "online"]
    filtered_transactions.drop(columns=["order_channel"], inplace=True)
    return filtered_transactions


def add_features(filtered_transactions):
    # Add unit sale price
    filtered_transactions["unit_price"] = filtered_transactions["sales_net"]/filtered_transactions["quantity"]
    
    # Add flow : 1 means it is a purchase (increased stock), -1 it is a sale
    filtered_transactions["stock_flow"] = filtered_transactions["sales_net"].apply(lambda x: 1 if x < 0 else -1)
    filtered_transactions["quantity"] = filtered_transactions["quantity"]*filtered_transactions["stock_flow"]
    
    # Transform dates
    filtered_transactions["date_order"] = pd.to_datetime(filtered_transactions["date_order"])
    filtered_transactions["date_invoice"] = pd.to_datetime(filtered_transactions["date_invoice"])
    
    # Date features
    filtered_transactions["month_order"] = filtered_transactions["date_order"].dt.month
    filtered_transactions["order_invoice_delta"] = (filtered_transactions["date_invoice"]- \
                                                    filtered_transactions["date_order"]).dt.days
    return filtered_transactions


def main():
    print("Loading raw data set...")
    raw_transactions = load_raw_transactions()
    
    print("Filtering transactions...")
    filtered_transactions = select_transactions(raw_transactions)
    
    print("Computing useful features...")
    filtered_transactions = add_features(filtered_transactions)
    
    print("Saving cleaned data set...")
    filtered_transactions.to_csv(os.path.join("data", "transactions_dataset_clean.csv"))
    return


if __name__=="__main__":
    main()
