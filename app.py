import os
from dotenv import load_dotenv
import pickle
import requests
import streamlit as st

# Load environment variables
load_dotenv()

def download_file_from_drive(file_id, destination):
    """Downloads a file from Google Drive."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """Gets the confirmation token for large files."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    """Saves the response content to a file."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# Download the PKL files
if not os.path.exists("movie_list.pkl"):
    download_file_from_drive("1V0BDie82HANz3tl5IALzOroYDuS7p8yc", "movie_list.pkl")

if not os.path.exists("similiarity.pkl"):
    download_file_from_drive("1JjK3tmDs3TZcXLlQBaKMNdOaTmn9mAav", "similiarity.pkl")

# Streamlit App Code
st.title('Movie Recommendation System')

def fetch_poster(movie_id): 
    api_key = os.getenv("API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US" 
    data = requests.get(url) 
    data = data.json() 
    poster_path = data.get('poster_path', '') 
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
        recommended_movies_poster.append(fetch_poster(i)) 
 
    return recommended_movies_name, recommended_movies_poster 

# Load movies and similarity data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similiarity = pickle.load(open('similiarity.pkl', 'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox('Type or select a movie to get recommendations:', movie_list)

if st.button('Show Recommendations'):
    recommended_movies_name, recommended_movies_poster = get_movie_recommendations(selected_movie, movies, similiarity)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(recommended_movies_name[i])
            st.image(recommended_movies_poster[i])
