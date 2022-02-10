import os
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


def load_raw_transactions():
    raw_transactions = pd.read_csv(os.path.join("data", "transactions_dataset.csv"), sep = ";")
    return raw_transactions


def select_transactions(raw_transactions):
    filtered_transactions = raw_transactions[raw_transactions.order_channel == "online"]
    filtered_transactions.drop(columns=["order_channel"], inplace=True)
    filtered_transactions.drop_duplicates(inplace=True)
    filtered_transactions.sort_values(by="date_order", inplace=True)
    return filtered_transactions


def add_date_features(filtered_transactions):
    # Transform dates
    filtered_transactions["date_order"] = pd.to_datetime(filtered_transactions["date_order"])
    filtered_transactions["date_invoice"] = pd.to_datetime(filtered_transactions["date_invoice"])
    
    filtered_transactions["order_invoice_delta"] = (filtered_transactions["date_invoice"] - \
                                                    filtered_transactions["date_order"]).dt.days
    return filtered_transactions


def client_specific_features(client_transactions):
    client_transactions.sort_values(by="date_order", inplace=True)
    client_transactions = client_transactions.groupby(by=["client_id", "branch_id", 
                                                          "date_order"]).agg({"order_invoice_delta":"max",
                                                                              "sales_net":"sum", 
                                                                              "quantity":"sum"})
    client_transactions.reset_index(inplace=True)
    
    client_transactions["past_lifetime_value"] = client_transactions.sales_net.cumsum()
    
    client_transactions["last_order"] = 0
    client_transactions["last_order"].iloc[-1] = 1
    
    client_transactions["time_since_last"] = client_transactions.date_order.diff().dt.days
    
    client_transactions["age"] = (client_transactions.date_order - client_transactions.date_order.iloc[0]).dt.days
    
    client_transactions["sales_net_growth_rate"] = round(client_transactions.sales_net.diff()/ \
                                                          client_transactions["sales_net"].shift(),
                                                     2)
    
    client_transactions["churned"] = client_transactions.date_order.apply(
        lambda x: int((client_transactions.date_order.max()-x).days > \
                  client_transactions.time_since_last.mean())) * client_transactions.last_order
    
    return client_transactions


def add_all_client_specific_features(filtered_transactions):
    ct = []
    for cid in filtered_transactions.client_id.unique():
        client_transactions = filtered_transactions[filtered_transactions.client_id == cid].loc[:,
                                              ["date_order", "order_invoice_delta", "branch_id", "client_id",  
                                               "sales_net", "quantity"]]
        client_transactions = client_specific_features(client_transactions)
        ct.append(client_transactions)
    return pd.concat(ct)


def main():
    print("Loading raw data set...")
    raw_transactions = load_raw_transactions()
    
    print("Filtering transactions...")
    # Remove this step to compare channels or work on a larger data set. Beware of high computationnal cost.
    filtered_transactions = select_transactions(raw_transactions)
    
    print("Computing useful features...")
    filtered_transactions = add_date_features(filtered_transactions)
    
    print("Computing client history features...")
    client_history = add_all_client_specific_features(filtered_transactions)
    # Remove clients with only one order, who have NaN for differential features
    client_history.dropna(inplace=True)
    
    print("Saving cleaned data set...")
    client_history.to_csv(os.path.join("data", "client_history.csv"))
    
    print("Preprocessing complete.")
    return


if __name__=="__main__":
    main()
    