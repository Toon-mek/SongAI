# Import necessary libraries
import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download the CSV from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Define genre keywords for prediction
genre_keywords = {
    'Rock': ['rock', 'guitar', 'band', 'drums'],
    'Pop': ['love', 'dance', 'hit', 'baby'],
    'Jazz': ['jazz', 'swing', 'blues', 'saxophone'],
    'Country': ['country', 'truck', 'road', 'cowboy'],
    'Hip Hop': ['rap', 'hip', 'hop', 'beat', 'flow'],
    'Classical': ['symphony', 'orchestra', 'classical', 'concerto']
}

# Function to predict genre based on keywords
def predict_genre(row):
    for genre, keywords in genre_keywords.items():
        text = f"{row['Song Title']} {row['Lyrics']}"
        if any(keyword.lower() in str(text).lower() for keyword in keywords):
            return genre
    return 'Unknown'

# Load emotion detection model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Detect emotions in lyrics
def detect_emotions(lyrics, emotion_model):
    max_length = 512
    truncated_lyrics = ' '.join(lyrics.split()[:max_length])
    emotions = emotion_model(truncated_lyrics)
    return emotions

# Compute similarity between songs based on lyrics
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

# Recommend similar songs
def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    song_lyrics = song_data['Lyrics'].values[0]
    song_genre = song_data['Predicted Genre'].values[0]
    
    emotion_model = load_emotion_model()
    detect_emotions(song_lyrics, emotion_model)  # Detect emotions (optional to use)
    
    similarity_scores = compute_similarity(df, song_lyrics)
    
    df['similarity'] = similarity_scores
    recommended_songs = df[(df['Predicted Genre'] == song_genre)].sort_values(by='similarity', ascending=False).head(top_n)
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'Predicted Genre', 'similarity']]

# Main function to run the app
def main():
    st.title("Song Recommender System Based on Lyrics Emotion and Genre")
    
    # Load the data
    data_df = download_data_from_drive()
    data_df['Predicted Genre'] = data_df.apply(predict_genre, axis=1)
    
    # Sidebar for genre selection
    st.sidebar.header('Filter Songs by Predicted Genre')
    unique_genres = data_df['Predicted Genre'].unique()
    unique_genres = [genre for genre in unique_genres if genre != 'Unknown']
    selected_genre = st.sidebar.selectbox('Select a Genre', options=['Select a genre'] + unique_genres)
    
    # Display songs filtered by selected genre
    if selected_genre != 'Select a genre':
        filtered_songs = data_df[data_df['Predicted Genre'] == selected_genre]
        filtered_songs['Release Date'] = pd.to_datetime(filtered_songs['Release Date'], errors='coerce')
        filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

        st.write(f"### Songs Filtered by Genre: {selected_genre}")
        for idx, row in filtered_songs.iterrows():
            with st.container():
                st.markdown(f"**No. {idx + 1}: {row['Song Title']}**")
                st.markdown(f"**Artist:** {row['Artist']}")
                st.markdown(f"**Album:** {row['Album']}")
                st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
                with st.expander("Show/Hide Lyrics"):
                    st.write(row['Lyrics'].strip())
                st.markdown("---")
    
    # Select a song for recommendations
    st.write("### Recommend Songs Similar to a Selected Song")
    song_list = data_df['Song Title'].unique()
    selected_song = st.selectbox("Select a Song", song_list)
    
    # Recommend songs when button is clicked
    if st.button("Recommend Similar Songs"):
        recommendations = recommend_songs(data_df, selected_song)
        st.write(f"### Recommended Songs Similar to {selected_song}")
        for idx, row in recommendations.iterrows():
            st.markdown(f"**No. {idx + 1}: {row['Song Title']}**")
            st.markdown(f"**Artist:** {row['Artist']}")
            st.markdown(f"**Album:** {row['Album']}")
            st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            st.markdown(f"**Genre:** {row['Predicted Genre']}")
            st.markdown(f"**Similarity Score:** {row['similarity']:.2f}")
            st.markdown("---")

# Run the app
if __name__ == '__main__':
    main()
