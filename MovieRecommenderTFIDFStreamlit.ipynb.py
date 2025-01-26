
#Importing relevant libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Assuming the CSV is in the same directory as your script or inside a folder like 'data/'
File1 = "data/tmdb_5000_movies.csv"
Movie = pd.read_csv(File1)
Movie.head()


#importing the file
File2 = R"C:\Users\CRYSTAL GAMING\Desktop\File A\Projects\Movie reccomendation system/tmdb_5000_credits.csv"
Credits = pd.read_csv(File2)
Credits.head()


#Merging both sheets
Movies = Movie.merge(Credits, on = 'title')


#Selecting relevant columns for analysis
Movies = Movies[['movie_id','title','overview','genres','keywords','cast','crew','vote_average','release_date']]


#Removing null values
Movies.dropna(inplace=True)


#Creating a function to extract top 3 unqiue rows
import ast

def convert(text):
    l = []
    for i in ast.literal_eval(text):
     l.append(i['name'])
    return l


#Applying the convert function to genre to extract the top 3 genre
Movies['genres'] =  Movies['genres'].apply(convert)
Movies.head()




#Applying the convert function to keywords to extract 
Movies['keywords'] =  Movies['keywords'].apply(convert)


#Extracting just the top 3 cast names
import ast

def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
         l.append(i['name'])
    return l



#Extracting just the top 3 cast names
Movies['cast'] =  Movies['cast'].apply(convert_cast)


#Extracting just the top 3 cast names
import ast

def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
         l.append(i['name'])
    return l


def fetch_director(text):
    l = []  # Initialize an empty list to store director names
    counter = 0  # Initialize a counter (not used in this case)
    for i in ast.literal_eval(text):  # Evaluate the string and iterate through the list of crew members
        if i['job'] == 'Director':  # Check if the crew member's job is 'Director'
            l.append(i['name'])  # Append the director's name to the list
            break  # Exit the loop after finding the first director
    return l  # Return the list of director names



# Apply the fetch_director function to extract director information from the 'crew' column
Movies['crew'] =  Movies['crew'].apply(fetch_director)


#splits the string x into a list of words, using spaces as the delimite
Movies['overview'] = Movies['overview'].apply(lambda x:x.split())
Movies.head()


def remove_space(word):
    l = []  # Initialize an empty list to store the modified characters
    for i in (word):  # Iterate through each character in the word
        l.append(i.replace(" ", ""))  # Remove spaces from each character
    return l  # Return the list of characters with no spaces


# Remove extra spaces from the 'cast', 'crew', 'genres', and 'keywords' columns
Movies['cast'] =  Movies['cast'].apply(remove_space)
Movies['crew'] =  Movies['crew'].apply(remove_space)
Movies['genres'] =  Movies['genres'].apply(remove_space)
Movies['keywords'] =  Movies['keywords'].apply(remove_space)


# Combine cast, crew, genres, keywords, overview, and release_date into a single string
Movies['tags'] = (
    Movies['cast'].astype(str) + " " +
    Movies['crew'].astype(str) + " " +
    Movies['genres'].astype(str) + " " +
    Movies['keywords'].astype(str) + " " +
    Movies['overview'].astype(str) + " " +
    Movies['release_date'].astype(str)
)

# Select specific columns to keep in the DataFrame
Movies = Movies[['movie_id','title','tags','vote_average','release_date']]  


# Join the list of tags into a single string for each movie
Movies['tags'] = Movies['tags'].apply(lambda x: " ".join(x))



# Fixing the 'tags' column in the DataFrame named 'movies'
Movies['tags'] = Movies['tags'].apply(lambda x: ''.join(x.split()))



#Converting all the letters to lower
Movies['tags'] = Movies['tags'].apply(lambda x:x.lower())
Movies.head()


# Import the spaCy library for natural language processing
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Step 1: Tokenize and display the tags column (before stop words removal)
Movies['tokens'] = Movies['tags'].apply(lambda x: [token.text for token in nlp(x)])
print("Tokens (Before Stop Words Removal):")
print(Movies[['title', 'tokens']].head())

# Step 2: Remove stop words and display the result
Movies['tokens_no_stopwords'] = Movies['tokens'].apply(lambda x: [token for token in x if not nlp.vocab[token].is_stop])
print("\nTokens (After Stop Words Removal):")
print(Movies[['title', 'tokens_no_stopwords']].head())

# Step 3: Reconstruct the tags without stop words and display the result
Movies['tags_no_stopwords'] = Movies['tokens_no_stopwords'].apply(lambda x: " ".join(x))
print("\nTags (After Reconstructing without Stop Words):")
print(Movies[['title', 'tags_no_stopwords']].head())



MoviesTFIDF = Movies[['movie_id','title','vote_average','tags_no_stopwords','release_date']]



# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Function to lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# Apply lemmatization to the 'tags_no_stopwords' column
MoviesTFIDF['tags_lemmatized'] = MoviesTFIDF['tags_no_stopwords'].apply(lemmatize_text)


# Drop the 'tags_no_stopwords' column from the DataFrame
MoviesTFIDF = MoviesTFIDF.drop(columns=['tags_no_stopwords'])


# Import TfidfVectorizer for converting text to numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer with a limit on max features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Retain the top 5000 terms based on TF-IDF scores

# Fit and transform the 'tags_no_stopwords' column
tfidf_matrix = tfidf_vectorizer.fit_transform(MoviesTFIDF['tags_lemmatized'])

# Display the shape of the reduced TF-IDF matrix
print(tfidf_matrix.shape)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the 'tags_no_stopwords' column to get the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(MoviesTFIDF['tags_lemmatized'])

# Step 3: Compute cosine similarity between the TF-IDF vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Create a DataFrame of cosine similarity (optional)
cosine_sim_df = pd.DataFrame(cosine_sim, index=MoviesTFIDF['movie_id'], columns=MoviesTFIDF['movie_id'])

# Display the cosine similarity matrix for the first few movies (for example)
cosine_sim_df.head()




def get_movie_recommendations(movie_title, MoviesTFIDF, cosine_sim_df, top_n=5):
    # Step 1: Check if the movie exists in the dataset
    if movie_title not in MoviesTFIDF['title'].values:
        return f"Error: Movie '{movie_title}' not found in the dataset."
    
    # Step 2: Find the movie_id for the given movie title
    movie_id = MoviesTFIDF[MoviesTFIDF['title'] == movie_title]['movie_id'].values[0]
    
    # Step 3: Check if movie_id exists in cosine_sim_df
    if movie_id not in cosine_sim_df.index:
        return f"Error: Movie with movie_id {movie_id} not found in the cosine similarity dataframe."
    
    # Step 4: Get the similarity scores for the input movie with all other movies
    similarity_scores = cosine_sim_df.loc[movie_id]
    
    # Step 5: Sort the movies based on similarity scores in descending order (highest first)
    similar_movies = similarity_scores.sort_values(ascending=False)
    
    # Step 6: Get the top N most similar movies (excluding the movie itself)
    recommended_movie_ids = similar_movies.iloc[1:top_n+1].index
    
    # Step 7: Retrieve the titles of the recommended movies based on movie_id
    recommended_movies = MoviesTFIDF[MoviesTFIDF['movie_id'].isin(recommended_movie_ids)][['title', 'vote_average']]
    
    # Step 8: Create a new DataFrame with only the required columns
    result_df = recommended_movies.rename(columns={'title': 'Movie Name', 'vote_average': 'release_date'})
    
    return result_df


# Example usage:
movie_title_input = "Interstellar"  # Input the title of the movie for recommendations
recommendations = get_movie_recommendations(movie_title_input, MoviesTFIDF, cosine_sim_df, top_n=5)
print(recommendations)



new_df = MoviesTFIDF #: Assign MoviesTFIDF to new_df
new_df.head()


import os
# Check if the directory 'MovieRecommender2' exists, create it if not
if not os.path.exists('MovieRecommender2'):
    os.makedirs('MovieRecommender2')


# Create the 'MovieRecommender' folder
os.makedirs('MovieRecommender', exist_ok=True)


# Import pickle for saving objects
import pickle

# Save the `new_df` DataFrame to a file
pickle.dump(new_df, open('MovieRecommender/movie_list.pkl', 'wb'))

# Save the `cosine_sim_df` DataFrame to a file
pickle.dump(cosine_sim_df, open('MovieRecommender/similiarity.pkl', 'wb'))
cosine_sim_df.shape



