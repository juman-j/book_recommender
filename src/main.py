'''
Main module

This module accumulates all the basic code of this application. 

The key steps are:
1) FastAPI initialization  
2) specification of all the functions in which the data preparation and transformation is performed
3) grouping of all preparatory functions into a pipeline and calling them
(the pipeline also contains a function with the algorithm for prediction)
'''
import chardet
import pandas as pd
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates

from .settings import settings

app = FastAPI()

templates = Jinja2Templates(directory="template")


@app.get(settings.main_url)
async def book_selection(request: Request):
    return templates.TemplateResponse("book_selection.html", {"request": request})


@app.post("/recommendations")
async def get_data(request: Request):
    from_data = await request.form()
    book_title = from_data.get('book_title')
    book_author = from_data.get('book_author')
    
    def read_csv_with_encoding(filename):
        """
        Read a CSV file with auto-detected encoding.

        Args:
            filename (str): The path to the CSV file.

        Returns:
            pandas.DataFrame: The DataFrame containing the data from the CSV file.
        """
        with open(filename, 'rb') as file:
            data = file.read()
        result = chardet.detect(data)
        encoding = result['encoding']
        df = pd.read_csv(filename, encoding=encoding, sep=';', on_bad_lines='warn')

        return df


    def preprocess_read_csv(filename):
        """
        Read a CSV file, handling different encodings.

        Args:
            filename (str): The path to the CSV file.

        Returns:
            pandas.DataFrame: The DataFrame.
        """
        try:
            df = pd.read_csv(filename, encoding='utf-8', sep=';', on_bad_lines='warn')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filename, encoding='latin-1', sep=';', on_bad_lines='warn')
            except UnicodeDecodeError:
                df = read_csv_with_encoding(filename)

        return df


    def preprocess_lowercase(df):
        """
        Args:
            df (DataFrame): The input dataset.

        Returns:
            DataFrame: The modified DataFrame with selected columns in lowercase.
        """
        df = df[df['Book-Rating'] != 0]

        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].str.lower()

        return df


    def preprocess_followers(df, book_title, book_author):
        """
        Args:
            df (DataFrame): The input dataset.
            book_title (str): The title of the book.
            book_author (str): The author of the book.
            
        Returns:
            numpy.ndarray: An array of unique user IDs.
        """
        readers = df['User-ID'][
            (df['Book-Title'] == book_title) &
            (df['Book-Author'].str.contains(book_author))]

        unique_readers = readers.unique()
        df = df[['User-ID', 'Book-Rating', 'Book-Title']]

        return unique_readers, df


    def preprocess_popular_books(df, followers):
        """
        Args:
            df (DataFrame): Original dataset.
            followers (numpy.ndarray): List of users who rated the selected book.
            
        Returns:
            DataFrame: A spreadsheet with user ratings for selected books.
        """
        # Selecting books rated by users who rated the selected book
        raw_data = df[(df['User-ID'].isin(followers))]

        # Books rated more than 7 times
        popular_books = raw_data.groupby('Book-Title').count().loc[lambda x: x['User-ID'] >= 8]
        # back to the dataframe form
        popular_books = raw_data[raw_data['Book-Title'].isin(popular_books.index)]

        dataset_for_corr = popular_books.pivot_table(
            index='User-ID',
            columns='Book-Title',
            values='Book-Rating'
            )

        return dataset_for_corr, popular_books


    def correlation(dataset_for_corr, popular_books, book_title):
        """
        Measuring the correlation between user ratings of a particular book 
        and the ratings of other books.

        Args:
            dataset_for_corr (pandas.DataFrame): Spreadsheet with user ratings for selected books.
            popular_books (pandas.DataFrame): Dataset with ratings from those people who
                                                rated our target book.

        Returns:
            result (pandas.DataFrame): DataFrame containing information about
                                        book titles, average ratings, and correlations.
        """
        book_titles = []
        correlations = []
        avgrating = []

        dataset_of_other_books = dataset_for_corr.drop(book_title, axis=1)

        for title in list(dataset_of_other_books.columns):
            book_titles.append(title.capitalize())
            correlations.append(round(
                dataset_for_corr[book_title].corr(dataset_of_other_books[title]),
                2))
            avgrating.append(round(
                (popular_books[popular_books['Book-Title'] == title].mean()).to_list()[1],
                2))

        tupels = list(zip(book_titles, avgrating, correlations))
        result = pd.DataFrame(
            tupels, columns=['book_title', 'rating', 'correlations']).sort_values(
                by='correlations',
                ascending=False)

        return result


    def process(df, book_title, book_author):
        """
        Preprocesses the input dataframe by applying several preprocessing steps.

        Args:
            df (pandas.DataFrame): The input dataframe to be preprocessed.
            book_title (str): The title of the book used to filter the dataframe.
            book_author (str): The author of the book used to filter the dataframe.

        Returns:
            DataFrame: A spreadsheet for measuring the correlation.
            DataFrame: A dataset of our target group's book scores.
        Steps:
            1. Lower case & eliminating zero rating: Encodes string columns to lowercase 
                for better consistency and comparison | Also eliminate zero ratings. 
            2. Preprocess Followers: Finds unique users who rated a particular book
                and filters the dataframe accordingly.
            3. Preprocess Popular Books: Identifies books with more than 7 ratings from 
                the target audience and creates a spreadsheet dataset.
            4. Correlation: Measuring the correlation between user ratings of a particular book
                and the ratings of other books.
        """
        book_title = book_title.lower()
        book_author = book_author.lower()
        
        df_preprocessed = df.copy()
        df_preprocessed = preprocess_lowercase(df_preprocessed)
        unique_readers, df_preprocessed = preprocess_followers(df_preprocessed, book_title, book_author)

        if unique_readers.size == 0:
            return None

        dataset_for_corr, popular_books = preprocess_popular_books(df_preprocessed, unique_readers)

        if dataset_for_corr.empty or popular_books.empty:
            return pd.DataFrame()

        result = correlation(dataset_for_corr, popular_books, book_title)

        return result


    ratings = preprocess_read_csv('data/BX-Book-Ratings.csv')
    books = preprocess_read_csv('data/BX-Books.csv')
    df = pd.merge(ratings, books, on=['ISBN'])

    recommendations = process(df, book_title, book_author)

    if recommendations is None:
        return templates.TemplateResponse("book_not_found.html", {"request": request})

    if recommendations.empty:
        return templates.TemplateResponse("no_recommendations.html", {"request": request})

    return templates.TemplateResponse("recommendations.html", {"request": request, "recommendations": recommendations.head(5)})

