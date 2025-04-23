#!/etc/bin/env python3
##
import time
from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from local_items.depreciated import predict

##
if __name__ == "__main__":
    ##
    freeze_support()
    rdp_data = Path("data/trainset19_072023_db.csv")
    print(f"ref db {rdp_data.name} fount: {rdp_data.exists()}")

    db = pd.read_csv(rdp_data)
    print(f"Size of the database: {db.shape}")

    X = db["sequence"]
    y = db['id']  # Target variable

    ##
    # Reload the module in case I edit the code
    classify = predict.Classify()
    classify.multi_processing = True

    start = time.time()
    classify.fit(db["sequence"],
                 db["id"],
                 multi=True,
                 n_cpu=6)

    end = time.time()
    print(f"Run time {(end - start):.1f} seconds")

    ## Split the data into training and testing sets
    # test_size=0.01 (1%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50)

    # Print the shapes of the resulting sets to verify the split
    print(f"Testing {X_test.shape} sequences")
    start = time.time()
    res = classify.predict(X_test, y_test, multi_p=False)
    end = time.time()
    print(f"Run time {(end - start):.1f} seconds")

    results_df = predict.summarize_predictions(res)

    ##
    accuracy = accuracy_score(results_df["id"], results_df["observed"])
    print(f"{accuracy:.2%}")
