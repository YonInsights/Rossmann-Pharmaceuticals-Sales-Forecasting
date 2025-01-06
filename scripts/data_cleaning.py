import os
import sys
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='missing_values.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_missing_values(df):
    # Calculate the percentage of missing values for each column
    missing_values = df.isnull().mean() * 100
    missing_values = missing_values[missing_values > 0]
    
    if missing_values.empty:
        logging.info("No missing values found.")
    else:
        logging.info("Columns with missing values and their percentage:")
        for column, percentage in missing_values.items():
            logging.info(f"{column}: {percentage:.2f}%")

def count_missing_values(train_data, test_data, store_data):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Log the start of the function
    logging.info('Starting count_missing_values function')

    # Count missing values in each column of the dataframes
    missing_values_train = train_data.isnull().sum()
    missing_values_test = test_data.isnull().sum()
    missing_values_store = store_data.isnull().sum()

    # Log the counts
    logging.info('Missing values in train data:\n%s', missing_values_train)
    logging.info('Missing values in test data:\n%s', missing_values_test)
    logging.info('Missing values in store data:\n%s', missing_values_store)

    # Print missing values for each dataframe
    print("Missing values in train data:\n", missing_values_train)
    print("\nMissing values in test data:\n", missing_values_test)
    print("\nMissing values in store data:\n", missing_values_store)

    # Log the end of the function
    logging.info('Finished count_missing_values function')
def replace_missing_values_with_zero(*dataframes):
    """
    Replace missing values with 0 in given dataframes and log the details.
    
    Parameters:
        *dataframes: list of pandas.DataFrame
            One or more pandas DataFrames to process.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    for df in dataframes:
        # Replace missing values with 0
        df.fillna(0, inplace=True)
        
        # Log verification
        logging.info(f"Missing values in dataframe:\n{df.isnull().sum()}")

def check_for_duplicates(*dataframes):
    """
    Check for duplicate rows in given dataframes and log the details.
    
    Parameters:
        *dataframes: list of pandas.DataFrame
            One or more pandas DataFrames to process.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    for df in dataframes:
        # Check for duplicates
        duplicates = df.duplicated().sum()
        
        # Log the result
        logging.info(f"Duplicate rows in dataframe: {duplicates}")
def preprocess_data(train_data, test_data):
    """
    Preprocess the train and test data by converting date fields to datetime format and 
    converting categorical variables to category types, while logging the process.
    
    Parameters:
        train_data (pandas.DataFrame): The training dataset.
        test_data (pandas.DataFrame): The testing dataset.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Convert date fields to datetime format
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    logging.info("Converted 'Date' fields to datetime format.")

    # Convert categorical variables to category types
    categorical_columns_train = ['Store', 'DayOfWeek', 'StateHoliday']
    categorical_columns_test = ['Store', 'DayOfWeek', 'StateHoliday']

    for col in categorical_columns_train:
        train_data[col] = train_data[col].astype('category')
    
    for col in categorical_columns_test:
        test_data[col] = test_data[col].astype('category')

    logging.info("Converted categorical variables to category types.")

    # Verify the changes
    logging.info(f"Data types in train data:\n{train_data.dtypes}")
    logging.info(f"Data types in test data:\n{test_data.dtypes}")





