import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire and Clean Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# ----------------------------------------------------------------- #
def get_zillow_sql():#this function acquires zillow data, and converts it to a dataframe

    ''' this function calls a sql file from the codeup database and creates a data frame from the zillow db.
    '''
    query ='''
        SELECT 
        bedroomcnt,
        bathroomcnt,
        calculatedfinishedsquarefeet,
        taxvaluedollarcnt,
        yearbuilt,
        taxamount,
        fips
        FROM
        properties_2017
            LEFT JOIN
        predictions_2017 USING (parcelid)
        join
            propertylandusetype USING (propertylandusetypeid)
            WHERE propertylandusedesc = 'Single Family Residential'
            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
            AND bathroomcnt >= 1 <= 6
            AND bedroomcnt >= 1 <= 6
        '''
    df = pd.read_sql(query, get_connection('zillow'))
    #creating a csv for easy access 
    return df
        
def get_zillow_df():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_zillow_sql()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df