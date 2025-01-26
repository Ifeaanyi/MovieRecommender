# Movie Recommendation System

## Overview
This repository contains a **movie recommendation system** built using **TF-IDF (Term Frequency-Inverse Document Frequency)** for text-based similarity and a **Streamlit** app for user interaction. The system provides movie recommendations based on the similarity of movie descriptions or metadata.

## Features
1. **Movie Similarity**:
   - Calculates similarity scores using TF-IDF for textual features.
   - Provides recommendations based on the highest similarity scores.
2. **Interactive Interface**:
   - Built with Streamlit for an intuitive user experience.
   - Users can input a movie title to get recommendations.
3. **Serialized Data**:
   - Movie codes and similarity scores are stored in `.pkl` files for faster loading and processing.
   
**app.py**:
Streamlit app that provides a user interface for movie recommendations.
Users can input a movie title to get recommendations, and the app fetches posters for each movie.

**.env File** 
  The .env file is used to store private information like API keys. Instead of putting your keys directly in the code, you save them in the .env file, which keeps them secure. You can then load these keys into your code using the python-dotenv package, so they don't show up in your public repositories.

## Files
- **`MovieRecommenderTfidStreamlit.ipynb`**:
  - Jupyter Notebook containing the implementation of the recommendation system.
  - Includes data preprocessing, TF-IDF calculation, and serialization of results.
- **`.pkl` Files** (not included in the repository):
  - `movie_codes.pkl`: Contains mapping of movie titles to codes.
  - `similarity_scores.pkl`: Stores the precomputed similarity scores matrix.

## How It Works
1. **Data Preprocessing**:
   - The dataset is cleaned and prepared, focusing on relevant text features (e.g., movie descriptions).
2. **TF-IDF Vectorization**:
   - Converts text data into numerical vectors using the TF-IDF method.
   - Measures the importance of words in each movie description.
3. **Similarity Calculation**:
   - Computes pairwise cosine similarity between movies based on their TF-IDF vectors.
   - Results are stored in a matrix for efficient querying.
4. **Streamlit App**:
   - Users input a movie title.
   - The app retrieves the most similar movies using the similarity matrix.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run MovieRecommenderApp.py
   ```

4. Add the `.pkl` files (if available) in the project directory or generate them using the Jupyter Notebook.

## Regenerating `.pkl` Files
If `.pkl` files are missing, regenerate them by running the notebook:
1. Open `MovieRecommenderTfidStreamlit.ipynb`.
2. Follow the steps to preprocess data, compute similarity, and serialize the files.

## Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- Streamlit
- pickle

## Notes
- Ensure the dataset is present and properly linked in the notebook.
- The `.pkl` files are excluded from the repository but can be generated as described above.

## Contributing
Feel free to fork this repository, raise issues, or submit pull requests to improve the system.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

