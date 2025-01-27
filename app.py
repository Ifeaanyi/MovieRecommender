import os  # Added for environment variable handling
from dotenv import load_dotenv  # Added for loading .env files
import pickle
import requests  # type: ignore # Ensure you have installed this module
import streamlit as st  # type: ignore

# Load environment variables from .env file
load_dotenv()

# Cache the movie list and similarity matrix loading
@st.cache_data
def load_movie_data():
    movies = pickle.load(open('Moviefiles/movie_list.pkl', 'rb'))
    similiarity = pickle.load(open('Moviefiles/similiarity.pkl', 'rb'))
    return movies, similiarity

# Cache the poster-fetching function
@st.cache_resource
def fetch_poster(movie_id): 
    # Fetch the API key from the .env file
    api_key = os.getenv("API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US" 
    data = requests.get(url) 
    data = data.json() 
    poster_path = data.get('poster_path', '')  # Use .get to avoid KeyError 
    if poster_path: 
        full_path = f"http://image.tmdb.org/t/p/w500/{poster_path}" 
    else: 
        full_path = "https://via.placeholder.com/500x750.png?text=No+Image" 
    return full_path
 
def get_movie_recommendations(movie_title, movies, cosine_sim_df, top_n=5): 
    if movie_title not in movies['title'].values: 
        return f"Error: Movie '{movie_title}' not found." 
 
    movie_id = movies[movies['title'] == movie_title]['movie_id'].values[0] 
    similarity_scores = cosine_sim_df.loc[movie_id] 
    similar_movies = similarity_scores.sort_values(ascending=False) 
    recommended_movie_ids = similar_movies.iloc[1:top_n + 1].index 
 
    recommended_movies_name = [] 
    recommended_movies_poster = [] 
 
    for i in recommended_movie_ids: 
        recommended_movies_name.append(movies[movies['movie_id'] == i]['title'].values[0]) 
        recommended_movies_poster.append(fetch_poster(i))  # Fetch poster for each movie 
 
    return recommended_movies_name, recommended_movies_poster 
 
# Streamlit App Code 
st.title('Movie Recommendation System') 
 
# Load movies and similarity data using cached function
movies, similiarity = load_movie_data()
 
# Get the list of movie titles for the select box 
movie_list = movies['title'].values 
selected_movie = st.selectbox('Type or select a movie to get recommendations:', movie_list) 
 
# Button to show recommendations
if st.button('Show Recommendations'): 
    recommended_movies_name, recommended_movies_poster = get_movie_recommendations(selected_movie, movies, similiarity) 
 
    # Display recommended movies in columns 
    cols = st.columns(5) 
    for i, col in enumerate(cols): 
        with col: 
            st.text(recommended_movies_name[i]) 
            st.image(recommended_movies_poster[i])
