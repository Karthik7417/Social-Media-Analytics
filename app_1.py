from dataCollection.elasticSearch_dataCollection import dataCollection
from dataCleaning.cleaning import cleaning
from algorithm.train_test import train_test_split
from dataCleaning.token_calculations import tokenization_ftm
from algorithm.logistic_regression import logistic_regression
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import time

# setting local time zone as New York (EST)
local = pytz.timezone("US/Eastern")


# Collect information for 10 days
def app(start_time, end_time):
    current_time = datetime.utcnow()
    if end_time > current_time:
        print("end_time given is greater than the the current UTC time so changing end_time to", current_time)
        end_time = current_time
    diff = end_time - start_time
    total_hours = int(np.ceil(diff.total_seconds() / 3600))
    summary_df = pd.DataFrame()
    trends_df = pd.DataFrame()
    print("Total time is", total_hours)

    for hours in range(0,total_hours,6):

        print("Data Collection for hours", hours, "/",total_hours)
        start_time, data, gv_scroll_size_total, non_gv_scroll_size_total, data_original = dataCollection(start_time)

        # Clean the initial dataset
        data = cleaning(data)

        # Split into train and test
        X, X_test, y, y_test = train_test_split(data)

        #Tokenization
        X, X_test = tokenization_ftm(X,X_test)


        # Converting y, y_test df to int
        y = y.astype('int')
        y_test = y_test.astype('int')

        print("Running Logistic regression")
        start = time.clock()
            # Run Logistic Regression
        trends_df, summary_df,prediction = logistic_regression(X,X_test,y,y_test,start_time,gv_scroll_size_total,
                                                               non_gv_scroll_size_total, trends_df, summary_df)
        print("Logistic Regression took", time.clock() - start, "seconds to run")


    summary_df.columns = ['datetime', 'gv_size', 'non_gv_size', 'tf_idf_size', 'f1_score', 'accuracy', 'precision',
                          'recall', 'train_size', 'test_size', 'TP', 'FP', 'TN', 'FN']

    trends_df.to_csv("reports/trends_df_" + str(start_time) + ".csv")
    summary_df.to_csv("reports/summary_df_" + str(start_time) + ".csv")


if __name__ == '__main__':
    app(datetime(2018, 12, 7, 0, 0, 0), datetime(2018, 12, 10, 0, 0, 0))
