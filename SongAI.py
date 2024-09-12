# Import necessary libraries
import pandas as pd
import streamlit as st
import gdown
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np

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
st.write(data_df)

# Data preprocessing: Encoding categorical data and handling missing values
label_encoder = LabelEncoder()
all_songs_data['genre'] = label_encoder.fit_transform(all_songs_data['genre'])
all_songs_data = all_songs_data.dropna()

# Split dataset into features and target variable
X = all_songs_data.drop(['song_name', 'artist_name', 'youtube_url'], axis=1)  # Features
y = all_songs_data['genre']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing a basic K-Nearest Neighbors algorithm for recommendations
model_knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
model_knn.fit(X_train)

# Streamlit application
st.title("Song Recommendation System")

# Sidebar for filtering songs by category
st.sidebar.header('Filter by Song Category')
category = st.sidebar.selectbox('Select Category', options=label_encoder.classes_)

# Filter songs based on selected category
filtered_songs = all_songs_data[all_songs_data['genre'] == label_encoder.transform([category])[0]]

# Display the filtered songs with an option to play the video
st.write(f"### Songs in {category} Category")
for index, row in filtered_songs.iterrows():
    st.write(f"**{row['song_name']}** by {row['artist_name']}")
    st.video(row['youtube_url'])

# Provide a sample recommendation when a song is selected
st.write("## Sample Recommendations")
sample_song_index = st.selectbox('Select a Song', filtered_songs.index)
sample_song = X.loc[sample_song_index].values.reshape(1, -1)
distances, indices = model_knn.kneighbors(sample_song)

# Display recommended songs based on KNN model
for idx in indices.flatten():
    song_name = all_songs_data.iloc[idx]['song_name']
    artist_name = all_songs_data.iloc[idx]['artist_name']
    youtube_url = all_songs_data.iloc[idx]['youtube_url']
    st.write(f"**{song_name}** by {artist_name}")
    st.video(youtube_url)

# Note: Model evaluation and accuracy can be expanded as needed based on specific requirements or metrics.
