import uuid

import pandas
import requests


# Read in a csv file from a given path
def get_data(filepath):
    df = pandas.read_csv(filepath)
    return df


# Replace all nan values in a column with a value or statistic
def replace_nan(df, col, is_percent=False, is_categorical=False):
    if df[col].isnull().values.any():
        print ("Removing NaNs from: ",col)
        if is_percent:
            df[col] = df[col].apply(percent_to_num)
            print (df[col].mean())
            df[col].fillna(df[col].mean(), inplace=True)
        if is_categorical:
            print (df[col].mode()[0])
            df[col].fillna(df[col].mode()[0], inplace=True)

#prints columns that contain NaNs
def nan_checker(df):
    count=0
    for col in list(df):
        if df[col].isnull().values.any():
            count += 1
            print("HAS NANS: ", col)
    if not count:
        print ("Columns are NaN free!!!!!!")

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
    return image_name


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


# Converts the string "{'a','b','c'}" to a real python list
def to_list(x):
    return x[1:-1].split(",")


# Reads in a csv file and replaces the nan values
df = get_data('data/1/listings_firststep.csv')
nan_checker(df)
try:
    replace_nan(df, 'host_response_rate', is_percent=True)
    replace_nan(df, 'host_acceptance_rate', is_percent=True)
    replace_nan(df, 'host_response_time', is_categorical=True)
except Exception as e:
    print (e)
nan_checker(df)
convert_to_columns(df,'host_response_time')
convert_to_columns(df, 'amenities')
print(list(df))

# Downloads all the images for a given column to the given dir
# df = get_data('data/3/listings_images_old.csv')
# download_images(df, 'picture_url', 'picture', 'data/3/listings_images.csv')
# print(df)
