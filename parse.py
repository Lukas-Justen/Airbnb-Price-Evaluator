import uuid

import pandas
import requests


# Read in a csv file from a given path
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


def get_data(filepath):
    df = pandas.read_csv(filepath)
    return df


# Replace all nan values in a column with a value or statistic
def replace_nan(df, col, is_percent=False, is_categorical=False):
    if df[col].isnull().values.any():
        # print("Removing NaNs from: ", col)
        if is_percent:
            df[col] = df[col].apply(percent_to_num)
            # print(df[col].mean())
            df[col].fillna(df[col].mean(), inplace=True)
        if is_categorical:
            # print(df[col].mode()[0])
            df[col].fillna(df[col].mode()[0], inplace=True)


# prints columns that contain NaNs
def nan_checker(df):
    count = 0
    for col in list(df):
        if df[col].isnull().values.any():
            count += 1
            # print("HAS NANS: ", col)
    if not count:
        print()
        # print("Columns are NaN free!!!!!!")


# Convert string to integer and return the integer
def percent_to_num(x):
    if isinstance(x, str):
        return int(x[:-1])
    else:
        return x


# Download all pictures for a given column and save new csv
def download_images(df, col, new_col, new_file):
    df[new_col] = df[col].apply(url_to_image)
    df.to_csv(new_file)


# Download the image of the url, save it and return its name
def url_to_image(url):
    output_dir = 'data/images/'
    image_name = uuid.uuid1().__str__() + '.jpg'
    file = open(output_dir + image_name, 'wb')
    image = requests.get(url).content
    file.write(image)
    file.close()
    return image_name


# Counts the length of a list in a column a creates a new column with the count
def count_list_in_column(df, col, new_name):
    df[new_name] = df[col].apply(lambda x: len(x.split(",")))
    del (df[col])


# Converts a column that contains a list to multiple columns with hotencoding
def convert_to_columns(df, col):
    values = get_distinct_values(df, col)
    if '' in values:
        values.remove('')
    for name in values:
        create_column(df, name, 0)
    for index, row in df.iterrows():
        columns = to_list(row[col])
        set_values(df, columns, index)
    del (df[col])


# Set the columns to 1 that are present in the root column
def set_values(df, columns, index):
    if '' in columns: columns.remove('')
    for column in columns:
        df.at[index, column] = 1


# Creates a new column with initial value in the dataframe
def create_column(df, name, initial_value):
    df[name] = initial_value


# Creates a set of unique value that appear within one column
def get_distinct_values(df, col):
    distinct_values = set()
    values = df[col].apply(to_list)
    for v in values:
        distinct_values |= set(v)
    return distinct_values


# Converts a column of string to hot encoded integers
def encode(df, col):
    keys = {x: i for i, x in enumerate(list(set(df[col])))}
    # print(keys)
    df[col] = df[col].map(keys)


# Converts the string "{'a','b','c'}" to a real python list
def to_list(x):
    return x[1:-1].lower().split(",")


# Converts the price column to an integer
def convert_price_to_integer(df, col):
    df[col] = df[col].apply(lambda x: float(x.replace('$', '').replace(',', '').replace('"', '')))


# Shuffles the rows of the given file and stores it back on disk
def shuffle_file(filename):
    df = pandas.read_csv(filename)
    df = df.sample(frac=1)
    df.to_csv(filename, index=False)


# Reduces the number of rows of the given file and stores it into anotehr file
def reduce_size(filename, rows, newfile):
    df = pandas.read_csv(filename)
    df = df.head(rows)
    df.to_csv(newfile, index=False)


# Add a new column to the given file with default value
def add_column_with(file, col, value):
    df = pandas.read_csv(file)
    df[col] = value
    df.to_csv(file)


# Concatenates all files in the given list
def csv_concat(filelist):
    df = pandas.DataFrame()
    for files in filelist:
        data = pandas.read_csv(files)
        if files == 'data/newyork/listings_details.csv':
            df = df.append(data[:3000])
        else:
            df = df.append(data)
    df.reset_index().drop(
        ['index','Unnamed: 0', 'host_since', 'host_id', 'host_name', 'id', 'market'],
        axis=1).to_csv('data/listings_first_concat.csv', index=False)

def convert_to_sentiment(df,col):
    df[col] = df[col].apply(convert_text_to_sentiment)

def convert_text_to_sentiment(text):
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

# function that returns cleaned dataframe
def get_processed_data():
    df = get_data('data/listings_first_concat.csv')

    nan_checker(df)
    try:
        replace_nan(df, 'host_response_rate', is_percent=True)
        replace_nan(df, 'host_acceptance_rate', is_percent=True)
        replace_nan(df, 'host_response_time', is_categorical=True)
        replace_nan(df, 'beds', is_categorical=True)
        replace_nan(df, 'review_scores_rating', is_percent=True)
    except Exception as e:
        print(e)
    nan_checker(df)
    print(df.columns)
    convert_price_to_integer(df, 'price')
    convert_to_columns(df, 'amenities')
    # count_list_in_column(df, 'amenities', "amenities_count")
    count_list_in_column(df, 'host_verifications', "verifications_count")
    encode(df, 'host_identity_verified')
    encode(df, 'host_response_time')
    # encode(df, 'market')
    encode(df, 'host_is_superhost')
    encode(df, 'property_type')
    encode(df, 'room_type')
    encode(df, 'bed_type')
    encode(df, 'cancellation_policy')
    encode(df, 'neighbourhood')
    convert_to_sentiment(df,'description')
    df.to_csv('data/listings_first_concat_clean.csv', index=False)
    return df

csv_concat(['data/boston/listings_details.csv', 'data/seattle/listings_details.csv','data/newyork/listings_details.csv'])
get_processed_data()