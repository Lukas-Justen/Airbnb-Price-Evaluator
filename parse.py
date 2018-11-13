import uuid

import pandas
import requests


# Read in a csv file from a given path
def get_data(filepath):
    df = pandas.read_csv(filepath)
    return df


# Replace all nan values in a column with a value or statistic
def replace_nan(df):
    for col in list(df):
        if df[col].isnull().values.any():
            print ("Removing NaNs from: ",col)
            if is_percent:
                df[col] = df[col].apply(percent_to_num)
            else:

            df[col].fillna(df[col].mean(), inplace=True)


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


# Reads in a csv file and replaces the nan values
df = get_data('data/1/listings_firststep.csv')
replace_nan(df, 'host_response_rate', is_percent=True)
replace_nan(df, 'host_acceptance_rate', is_percent=True)
print(df)

# Downloads all the images for a given column to the given dir
# df = get_data('data/3/listings_images_old.csv')
# download_images(df, 'picture_url', 'picture', 'data/3/listings_images.csv')
# print(df)
