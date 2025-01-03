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

