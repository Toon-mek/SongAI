# Import necessary libraries
import pandas as pd
import streamlit as st
import gdown
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np
from textblob import TextBlob

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

# Display the data in Streamlit
st.write("Original Dataset:")
st.write(data_df.head())

# Function to analyze the emotion of the lyrics
def analyze_lyrics_emotion(lyrics):
    analysis = TextBlob(str(lyrics))  # Convert lyrics to string to handle any potential non-string entries
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Add a new column for sentiment analysis of the lyrics
data_df['Emotion'] = data_df['Lyrics'].apply(analyze_lyrics_emotion)

# Display the dataset with the new 'Emotion' column
st.write("Dataset with Emotion Analysis:")
st.write(data_df[['Song Title', 'Artist', 'Lyrics', 'Emotion']])

# Data preprocessing: Encoding categorical data and handling missing values
label_encoder = LabelEncoder()
data_df['genre'] = label_encoder.fit_transform(data_df['Year'].fillna('Unknown'))  # Encoding 'Year' for genre-like categorization
data_df = data_df.dropna(subset=['Lyrics'])  # Drop rows where 'Lyrics' is missing

# Split dataset into features and target variable
X = data_df.drop(['Song Title', 'Artist', 'Lyrics', 'Album URL', 'Media', 'Song URL', 'Writers', 'Emotion'], axis=1, errors='ignore')  # Features
y = data_df['genre']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing a basic K-Nearest Neighbors algorithm for recommendations
model_knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
model_knn.fit(X_train)

# Streamlit application
st.title("Song Recommendation System with Emotion Analysis")

# Sidebar for filtering songs by category
st.sidebar.header('Filter by Song Category')
category = st.sidebar.selectbox('Select Category', options=label_encoder.classes_)

# Filter songs based on selected category
filtered_songs = data_df[data_df['genre'] == label_encoder.transform([category])[0]]

# Display the filtered songs with an option to play the video
st.write(f"### Songs in {category} Category")
for index, row in filtered_songs.iterrows():
    st.write(f"**{row['Song Title']}** by {row['Artist']}")
    st.write(f"**Emotion**: {row['Emotion']}")
    if pd.notna(row['Media']) and 'youtube' in str(row['Media']):
        for media in eval(row['Media']):
            if media['provider'] == 'youtube':
                st.video(media['url'])

# Provide a sample recommendation when a song is selected
st.write("## Sample Recommendations")
sample_song_index = st.selectbox('Select a Song', filtered_songs.index)
sample_song = X.loc[sample_song_index].values.reshape(1, -1)
distances, indices = model_knn.kneighbors(sample_song)

# Display recommended songs based on KNN model
for idx in indices.flatten():
    song_name = data_df.iloc[idx]['Song Title']
    artist_name = data_df.iloc[idx]['Artist']
    youtube_url = [media['url'] for media in eval(data_df.iloc[idx]['Media']) if media['provider'] == 'youtube'][0]
    st.write(f"**{song_name}** by {artist_name}")
    st.video(youtube_url)
