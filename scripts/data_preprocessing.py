import logging
import pandas as pd

def detect_and_handle_outliers(data, columns):
    """
    Detect and handle outliers in specified columns using the IQR method, log the details, and show the outliers.
    
    Parameters:
        data (pandas.DataFrame): The dataset.
        columns (list): List of columns to detect and handle outliers in.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    def cap_outliers(data, column, lower_bound, upper_bound):
        data[column] = data[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)

    for column in columns:
        outliers, lower_bound, upper_bound = detect_outliers_iqr(data, column)
        
        # Log detected outliers
        logging.info(f"Outliers in {column}:\n{outliers}")

        # Display the outliers
        print(f"Outliers in {column}:\n{outliers}")

        # Cap outliers
        cap_outliers(data, column, lower_bound, upper_bound)

        # Log the capping process
        logging.info(f"After capping outliers in {column}:\n{data[column].describe()}")

def create_new_features(train_data, test_data):
    """
    Create new features for train and test data, and log the process.
    
    Parameters:
        train_data (pandas.DataFrame): The training dataset.
        test_data (pandas.DataFrame): The testing dataset.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Create new features for train data
    train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
    train_data['Month'] = train_data['Date'].dt.month
    train_data['Year'] = train_data['Date'].dt.year
    train_data['DaysSinceLastPromo'] = (train_data['Date'] - train_data['Date'][train_data['Promo'] == 1].max()).dt.days
    
    logging.info("Created new features for train data.")

    # Create new features for test data
    test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek
    test_data['Month'] = test_data['Date'].dt.month
    test_data['Year'] = test_data['Date'].dt.year
    test_data['DaysSinceLastPromo'] = (test_data['Date'] - test_data['Date'][test_data['Promo'] == 1].max()).dt.days
    
    logging.info("Created new features for test data.")

    # Log the first few rows of the train data to verify the new features
    logging.info(f"First few rows of train data:\n{train_data.head()}")

