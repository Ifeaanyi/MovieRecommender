import pickle 
import requests  # Missing import 
import streamlit as st 

def fetch_poster(movie_id): 
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=78bd5238483b14cf120d515496b437eb&language=en-US" 
        data = requests.get(url)
        data.raise_for_status()  # Raises an exception for HTTP errors
        data = data.json() 
        poster_path = data.get('poster_path', '')  # Use .get to avoid KeyError 
        if poster_path:
            full_path = f"http://image.tmdb.org/t/p/w500/{poster_path}" 
        else:
            full_path = "https://via.placeholder.com/500x750.png?text=No+Image" 
        return full_path
    except requests.exceptions.RequestException as e:
        # Handle HTTP-related errors like network issues or invalid responses
        print(f"Error fetching poster for movie_id {movie_id}: {e}")
        return "https://via.placeholder.com/500x750.png?text=Error+Loading+Image"
    except Exception as e:
        # Handle other errors
        print(f"Unexpected error: {e}")
        return "https://via.placeholder.com/500x750.png?text=Error+Loading+Image"

def get_movie_recommendations(movie_title, movies, cosine_sim_df, top_n=5):
    try:
        if movie_title not in movies['title'].values:
            return f"Error: Movie '{movie_title}' not found.", []

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

    except KeyError as e:
        # Handle case where the movie_id or other keys are missing
        print(f"Error: Missing key {e}")
        return "Error in fetching movie recommendations", []
    except Exception as e:
        # Handle other errors
        print(f"Unexpected error: {e}")
        return "Unexpected error occurred", []

# Streamlit App Code 
st.title('Movie Recommendation System')

# Load movies and similarity data 
try:
    movies = pickle.load(open('Moviefiles/movie_list.pkl', 'rb'))
    similiarity = pickle.load(open('Moviefiles/similiarity.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error: File not found. Please ensure the movie list and similarity files exist.")
    print(f"Error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    print(f"Unexpected error: {e}")
    st.stop()

# Get the list of movie titles for the select box 
movie_list = movies['title'].values
selected_movie = st.selectbox('Type or select a movie to get recommendations:', movie_list)

# Button to show recommendations 
if st.button('Show Recommendations'):
    recommended_movies_name, recommended_movies_poster = get_movie_recommendations(selected_movie, movies, similiarity)

    if isinstance(recommended_movies_name, str):  # If an error message is returned
        st.error(recommended_movies_name)
    else:
        # Display recommended movies in columns 
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(recommended_movies_name[i])
                st.image(recommended_movies_poster[i])
