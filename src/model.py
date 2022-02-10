import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def load_client_history():
    client_history = pd.read_csv(os.path.join("../data", "client_history.csv")).drop(columns="Unnamed: 0")
    client_history["date_order"] = pd.to_datetime(client_history["date_order"])
    return client_history


def aggregate_client_history_features(client_history):
    clients = client_history.drop(columns=['quantity', 'last_order', 'date_order']).groupby(
                              by=["client_id", "branch_id"]).agg(max_delay = ('order_invoice_delta','max'),
                                                                   average_sales_net = ('sales_net','mean'), 
                                                                   lifetime_value = ('past_lifetime_value','max'), 
                                                                   avg_time_between_orders = ('time_since_last','mean'), 
                                                                   lifespan_days = ('age','max'), 
                                                                   avg_growth_rate = ('sales_net_growth_rate','mean') , 
                                                                   churned = ('churned','max'))
    clients.reset_index(inplace=True)
    clients.replace([np.inf, -np.inf], np.nan, inplace=True)
    clients.dropna(inplace=True)
    return clients


def prepare_train_test_data(clients):
    X = clients.set_index("client_id").drop(columns=["churned"])
    y = clients.set_index("client_id").loc[:,["churned"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print("Train size : ", X_train.shape[0])
    print("Test size : ", X_test.shape[0])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return  X_train, X_test, y_train, y_test


def main():
    print("Loading client history...")
    client_history = load_client_history()
    
    print("Preparing training data...")
    clients = aggregate_client_history_features(client_history)
    X_train, X_test, y_train, y_test = prepare_train_test_data(clients)
    
    print("Training GradientBoosting model...")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    print("Model accuracy : ", round(model.score(X_test, y_test), 1))
    y_pred = model.predict(X_test) 
    cm = confusion_matrix(y_test, y_pred)
    
    print("Model confusion matrix : ")
    print(cm)
    print("Model recall : ")
    print("Model precision : ")
    
    return


if __name__ == "__main__":
    main()
  