import streamlit as st
import pandas as pd
import gdown
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download the CSV from Google Drive
@st.cache_data
def download_data_from_drive():
    # Google Drive link for the dataset (convert to direct download link)
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'  # Replace FILE_ID with the actual file ID
    output = 'songTest1.csv'  # Change to the desired output file name
    
    # Download the file without printing progress (quiet=True)
    gdown.download(url, output, quiet=True)
    
    # Load the dataset
    return pd.read_csv(output)

# Load the dataset of your CSV file
data_df = download_data_from_drive()

# Define a dictionary with genre keywords
genre_keywords = {
    'Rock': ['rock', 'guitar', 'band', 'drums'],
    'Pop': ['love', 'dance', 'hit', 'baby'],
    'Jazz': ['jazz', 'swing', 'blues', 'saxophone'],
    'Country': ['country', 'truck', 'road', 'cowboy'],
    'Hip Hop': ['rap', 'hip', 'hop', 'beat', 'flow'],
    'Classical': ['symphony', 'orchestra', 'classical', 'concerto']
}

# Function to predict genre based on keywords in song title or lyrics
def predict_genre(row):
    for genre, keywords in genre_keywords.items():
        text = f"{row['Song Title']} {row['Lyrics']}"  # Combine relevant text fields
        if any(keyword.lower() in str(text).lower() for keyword in keywords):
            return genre
    return 'Unknown'  # Default if no keywords are matched

def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotions(lyrics, emotion_model):
    # Truncate lyrics to a maximum length (e.g., 512 tokens)
    max_length = 512
    truncated_lyrics = ' '.join(lyrics.split()[:max_length])
    emotions = emotion_model(truncated_lyrics)
    return emotions
    
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

def recommend_songs(df, selected_song, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    song_lyrics = song_data['Lyrics'].values[0]
    song_genre = song_data['Predicted Genre'].values[0]
    
    emotion_model = load_emotion_model()
    song_emotion = detect_emotions(song_lyrics, emotion_model)
    
    similarity_scores = compute_similarity(df, song_lyrics)
    
    df['similarity'] = similarity_scores
    recommended_songs = df[(df['Predicted Genre'] == song_genre)].sort_values(by='similarity', ascending=False).head(top_n)
        # Sort the filtered songs by 'Release Date' in descending order
    filtered_songs['Release Date'] = pd.to_datetime(filtered_songs['Release Date'], errors='coerce')  # Convert to datetime
    filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

def main():

    # Display each song in a banner format with an expander to show/hide lyrics
    st.write(f"### Playlist: {selected_genre}")
    for idx, row in filtered_songs.iterrows():
        with st.container():
            # Combine the song number and title into a single line
            st.markdown(f"**No. {idx + 1}: {row['Song Title']}**")
            st.markdown(f"**Artist:** {row['Artist']}")
            st.markdown(f"**Album:** {row['Album']}")
            st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            
            # Use expander to show/hide lyrics
            with st.expander("Show/Hide Lyrics"):
                st.write(row['Lyrics'].strip())  # Clean up the lyrics display
            st.markdown("---")  # Separator between songs

if _name_ == '_main_':
    main()
