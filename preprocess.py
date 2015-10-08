""" GjC 2015 kaggle: Rossmann """
""" Simple script to extract promo interval as features """

import pandas as pd
import numpy as np
from pandas import merge
from sklearn.feature_extraction import DictVectorizer

def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def normal(df, cols):
    """ Convert to standard normal """
    df = df._get_numeric_data()
    for col in cols:
        diff = df[col].max() - df[col].min()
        if diff:
            df[col] = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())
        else:
            df[col] = 0
    return df

def process_data(data, write_path, remove_zero_sales=True):
    """ Cleanup data """
    
    """ Create vector of promo intervals """
    p_month = ['p_jan', 'p_feb', 'p_mar',\
            'p_apr', 'p_may', 'p_jun',\
            'p_jul', 'p_aug', 'p_sep',\
            'p_oct', 'p_nov', 'p_dec']
    mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May',\
            'Jun', 'Jul', 'Aug', 'Sept', 'Oct',\
            'Nov', 'Dec']

    for month in p_month:
        data[month] = 0

    data.PromoInterval = data.PromoInterval.fillna('Void')

    for i in range(0, 12):
        data[p_month[i]] = data.PromoInterval.apply(lambda x: 1 if mon[i] in x.split(",") else 0)

    data.drop(['PromoInterval'], axis=1, inplace=True)
    
    """ Split date into parts """
    data['Year'] = data.Date.apply(lambda x: x.year)
    data['Month'] = data.Date.apply(lambda x: x.month)
    data['Woy'] = data.Date.apply(lambda x: x.weekofyear)

    """ Remove nans """
    data.fillna(0, inplace=True)
    
    """ Remove date and stateholiday """
    if 'Customers' in data:
        data.drop(['Customers'], axis=1, inplace=True)    
    data.drop(['Date', 'StateHoliday'], axis=1, inplace=True)

    """ Encode categorical data """
    data = encode_onehot(data, ['StoreType', 'Assortment'])

    """ Normalize everything"""
    data = data.convert_objects(convert_numeric=True)
    data = normal(data, ['Store', 'DayOfWeek', 'Promo2SinceYear', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Year', 'Month', 'Woy'])

    """ Filter data to dates where shop is open and non zero sales """
    if remove_zero_sales:
        if 'Sales' in data:
            data = data[data['Sales']!=0] 
        data = data[data['Open']==1]

    """ Write the data """
    if write_path:
        data.to_csv(write_path, index=False)
    return data


def main():
    """ Main part of program """
    infile = 'data/store.csv'
    outfile = 'data/stores_feat.csv'
    training_file = 'data/train.csv'
    test_file = 'data/test.csv'
    training_vector = 'data/training_vector.csv'
    test_vector = 'data/test_vector.csv'
    stores = pd.read_csv(infile, dtype=object)

    training_data = pd.read_csv(training_file, parse_dates = ['Date'], dtype=object)
    test_data = pd.read_csv(test_file, parse_dates=['Date'], dtype=object)

    training_data = merge(training_data, stores)
    test_data = merge(test_data, stores)

    process_data(training_data, training_vector)
    process_data(test_data, test_vector, remove_zero_sales=False)

if __name__ == "__main__":
    main()