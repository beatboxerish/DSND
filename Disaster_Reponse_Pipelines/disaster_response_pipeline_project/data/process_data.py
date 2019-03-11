import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, sep = ',')
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    category_colnames = categories.iloc[0,:].str.replace(r'\-\d$',\
'',regex = True)
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    
    df.drop('categories',axis = 1, inplace = True)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_messages_clean', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'data/disaster_messages.csv data/disaster_categories.csv '\
              'data/DisasterResponse.db')


if __name__ == '__main__':
    main()
