import pandas
from pandas import DataFrame


#function to read in data from a given filepath
#returns a dataframe
def get_data(filepath):
    df = pandas.read_csv(filepath)
    return df

#replaces a nan value in a column with a value/statistic. Currently is the mean of the column. Provision for converting string percentages
#to ints
#doesn't return anything, performs in place replacement
def replace_nan(df, col, is_percent = False):
    if is_percent:
        df[col] = df[col].apply(percent_to_num)
    df[col].fillna(df[col].mean(), inplace=True)

def percent_to_num(x):
    if isinstance(x,str):
        return int(x[:-1])
    else:
        return x


df = get_data('data/1/listings_firststep.csv')
replace_nan(df,'host_response_rate', is_percent=True)
replace_nan(df,'host_acceptance_rate', is_percent=True)
print (df)
