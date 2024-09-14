import streamlit as st
import pandas as pd
import gdown
import ast
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the data from Google Drive
@st.cache_data
def download_data_from_drive():
    url = 'https://drive.google.com/uc?id=1Woi9GqjiQE7KWIem_7ICrjXfOpuTyUL_'
    output = 'songTest1.csv'
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

# Load emotion detection model and tokenizer once
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("text-classification", model=model_name, top_k=None)
    return model, tokenizer

# Detect emotions in the song lyrics
def detect_emotions(lyrics, _emotion_model, _tokenizer):
    if not isinstance(lyrics, str):
        return []  # Return empty if lyrics are not a valid string
    max_length = 512  # Max token length for the model
    try:
        emotions = _emotion_model(lyrics[:max_length])  # Truncate to max length
        return emotions
    except Exception as e:
        st.write(f"Error in emotion detection: {e}")
        return []

# Compute similarity between the input song lyrics and all other songs in the dataset
@st.cache_data
def compute_similarity(df, song_lyrics):
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)  # Ensure all lyrics are strings
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Lyrics'])
    song_tfidf = vectorizer.transform([song_lyrics])
    similarity_scores = cosine_similarity(song_tfidf, tfidf_matrix)
    return similarity_scores.flatten()

def extract_youtube_url(media_str):
    """Extract the YouTube URL from the Media field."""
    try:
        media_list = ast.literal_eval(media_str)  # Safely evaluate the string to a list
        for media in media_list:
            if media.get('provider') == 'youtube':
                return media.get('url')
    except (ValueError, SyntaxError):
        return None

# Recommend similar songs based on lyrics and detected emotions
def recommend_songs(df, selected_song, selected_emotion=None, top_n=5):
    song_data = df[df['Song Title'] == selected_song]
    if song_data.empty:
        st.write("Song not found.")
        return []
    
    song_lyrics = song_data['Lyrics'].values[0]

    # Load emotion detection model and tokenizer
    emotion_model, tokenizer = load_emotion_model()

    # Detect emotions in the selected song
    emotions = detect_emotions(song_lyrics, emotion_model, tokenizer)
    st.write(f"### Detected Emotions in {selected_song}:")
    
    if emotions and len(emotions) > 0:
        # Extract the emotions list from the first item
        emotion_list = emotions[0]
        
        # Find the emotion with the highest score
        if isinstance(emotion_list, list) and len(emotion_list) > 0:
            top_emotion = max(emotion_list, key=lambda x: x['score'])
            emotion_sentence = f"The emotion of the song is **{top_emotion['label']}**."
        else:
            emotion_sentence = "No emotions detected."
        
        st.write(emotion_sentence)
    else:
        st.write("No emotions detected.")

    # Filter songs by the selected emotion if specified
    if selected_emotion:
        emotion_filtered_rows = []
        for idx, row in df.iterrows():
            if isinstance(row['Lyrics'], str):  # Check if lyrics are valid
                detected_emotions = detect_emotions(row['Lyrics'], emotion_model, tokenizer)
                if detected_emotions:
                    detected_emotion = max(detected_emotions[0], key=lambda x: x['score'])['label']
                    if detected_emotion.lower() == selected_emotion.lower():
                        emotion_filtered_rows.append(row)
        emotion_filtered_df = pd.DataFrame(emotion_filtered_rows)
    else:
        emotion_filtered_df = df

    # Check if any songs match the selected emotion
    if emotion_filtered_df.empty:
        st.write(f"No songs found with the emotion: {selected_emotion}.")
        return []

    # Compute lyrics similarity
    similarity_scores = compute_similarity(emotion_filtered_df, song_lyrics)

    # Add similarity scores to the dataframe
    emotion_filtered_df['similarity'] = similarity_scores

    # Exclude the selected song from recommendations
    emotion_filtered_df = emotion_filtered_df[emotion_filtered_df['Song Title'] != selected_song]

    # Recommend top N similar songs
    recommended_songs = emotion_filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommended_songs[['Song Title', 'Artist', 'Album', 'Release Date', 'similarity', 'Song URL', 'Media']]

def display_random_songs(df, n=5):
    random_songs = df.sample(n=n)
    st.write("### Discover Songs:")
    for idx, row in random_songs.iterrows():
        youtube_url = extract_youtube_url(row.get('Media', ''))
        if youtube_url:
            # If a YouTube URL is available, make the song title a clickable hyperlink
            song_title = f"<a href='{youtube_url}' target='_blank' style='color: #FA8072; font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</a>"
        else:
            # If no YouTube URL, just display the song title
            song_title = f"<span style='font-weight: bold; font-size: 1.2rem;'>{row['Song Title']}</span>"

        with st.container():
            st.markdown(song_title, unsafe_allow_html=True)
            st.markdown(f"**Artist:** {row['Artist']}")
            st.markdown(f"**Album:** {row['Album']}")
            st.markdown(f"**Release Date:** {row['Release Date'].strftime('%Y-%m-%d') if pd.notna(row['Release Date']) else 'Unknown'}")
            st.markdown("---")

def main():
    # Add custom CSS to change the background image
    st.markdown(
        """
        <style>
        .main {
            background-image: url('https://wallpapercave.com/wp/wp11163687.jpg');
            background-size: cover;
            background-position: center;
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: black;
            font-weight: 700;
            text-align: center;
        }
        .stButton>button {
            background-color: #fa8072;
            color: white;
            border-radius: 10px;
        }
        .stTextInput input {
            border: 1px solid #fa8072;
            padding: 0.5rem;
        }
        .stTextInput label {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🎵 Song Recommender Based on Lyrics & Emotions 🎶")
    df = download_data_from_drive()

    # Drop duplicate entries based on 'Song Title', 'Artist', 'Album', and 'Release Date'
    df = df.drop_duplicates(subset=['Song Title', 'Artist', 'Album', 'Release Date'], keep='first')

    # Convert the 'Release Date' column to datetime if possible
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

    # Ensure all lyrics are strings
    df['Lyrics'] = df['Lyrics'].fillna('').astype(str)

    # Option selection between search or emotion
    search_or_emotion = st.radio("Choose an option:", ("Search for a Song or Artist 🎤", "Select an Emotion Category 🎭"))

    if search_or_emotion == "Search for a Song or Artist 🎤":
        # Search bar for song name or artist
        search_term = st.text_input("Enter a Song Name or Artist").strip()

        if search_term:
            # Filter by song title or artist name
            filtered_songs = df[
                (df['Song Title'].str.contains(search_term, case=False, na=False)) |
                (df['Artist'].str.contains(search_term, case=False, na=False))
            ]

            filtered_songs = filtered_songs.sort_values(by='Release Date', ascending=False).reset_index(drop=True)

            if filtered_songs.empty:
                st.write("No songs found matching the search term.")
            else:
                st.write(f"### Search Results for: {search_term}")
                for idx, row in filtered_songs.iterrows():
                    with st.container():
                        st.markdown(f"<h2 style='font-weight: bold;'> {idx + 1}. {row['Song Title']}**</h2>", unsafe_allow_html=True)
                        st.markdown(f"*Artist:* {row['Artist']}")
                        st.markdown(f"*Album:* {row['Album']}")

                        if pd.notna(row['Release Date']):
                            st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d')}")
                        else:
                            st.markdown(f"*Release Date:* Unknown")

                        song_url = row.get('Song URL', '')
                        if pd.notna(song_url) and song_url:
                            st.markdown(f"[View Lyrics on Genius]({song_url})")

                        youtube_url = extract_youtube_url(row.get('Media', ''))
                        if youtube_url:
                            video_id = youtube_url.split('watch?v=')[-1]
                            st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                        with st.expander("Show/Hide Lyrics"):
                            lyrics = str(row['Lyrics']).strip().replace('\n', '\n\n') if isinstance(row['Lyrics'], str) else "Lyrics not available."
                            st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{lyrics}</pre>", unsafe_allow_html=True)
                        st.markdown("---")

                song_list = filtered_songs['Song Title'].unique()
                selected_song = st.selectbox("Select a Song for Recommendations 🎧", song_list)

                if st.button("Recommend Similar Songs"):
                    recommendations = recommend_songs(df, selected_song)
                    st.write(f"### Recommended Songs Similar to {selected_song}")
                    
                    for idx, row in enumerate(recommendations.iterrows(), 1):
                        st.markdown(f"<h2 style='font-weight: bold;'> {idx}. {row[1]['Song Title']}</h2>", unsafe_allow_html=True)
                        st.markdown(f"*Artist:* {row[1]['Artist']}")
                        st.markdown(f"*Album:* {row[1]['Album']}")

                        if pd.notna(row[1]['Release Date']):
                            st.markdown(f"*Release Date:* {row[1]['Release Date'].strftime('%Y-%m-%d')}")
                        else:
                            st.markdown(f"*Release Date:* Unknown")

                        st.markdown(f"*Similarity Score:* {row[1]['similarity']:.2f}")

                        youtube_url = extract_youtube_url(row[1].get('Media', ''))
                        if youtube_url:
                            video_id = youtube_url.split('watch?v=')[-1]
                            st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                        st.markdown("---")

    elif search_or_emotion == "Select an Emotion Category 🎭":
        # Emotion selection dropdown
        emotion_options = ['Happy', 'Sad', 'Anger', 'Fear']
        selected_emotion = st.selectbox("Select an Emotion", emotion_options)

        if st.button("Show Songs for Selected Emotion"):
            # Filter songs by selected emotion and display only top 5
            emotion_filtered_rows = []
            emotion_model, tokenizer = load_emotion_model()
            
            for idx, row in df.iterrows():
                if isinstance(row['Lyrics'], str):  # Ensure lyrics are a valid string
                    detected_emotions = detect_emotions(row['Lyrics'], emotion_model, tokenizer)
                    if detected_emotions:
                        detected_emotion = max(detected_emotions[0], key=lambda x: x['score'])['label']
                        st.write(f"Detected emotion for {row['Song Title']}: {detected_emotion}")  # Debugging output
                        if detected_emotion.lower() == selected_emotion.lower():
                            emotion_filtered_rows.append(row)
            
            emotion_filtered_df = pd.DataFrame(emotion_filtered_rows)

            # Show only top 5 songs based on release date or another criteria if needed
            emotion_filtered_df = emotion_filtered_df.sort_values(by='Release Date', ascending=False).head(5)

            if emotion_filtered_df.empty:
                st.write(f"No songs found with the emotion: {selected_emotion}.")
            else:
                st.write(f"### Top 5 Songs with Emotion: {selected_emotion}")
                for idx, row in emotion_filtered_df.iterrows():
                    with st.container():
                        st.markdown(f"<h2 style='font-weight: bold;'> {idx + 1}. {row['Song Title']}</h2>", unsafe_allow_html=True)
                        st.markdown(f"*Artist:* {row['Artist']}")
                        st.markdown(f"*Album:* {row['Album']}")

                        if pd.notna(row['Release Date']):
                            st.markdown(f"*Release Date:* {row['Release Date'].strftime('%Y-%m-%d')}")
                        else:
                            st.markdown(f"*Release Date:* Unknown")

                        song_url = row.get('Song URL', '')
                        if pd.notna(song_url) and song_url:
                            st.markdown(f"[View Lyrics on Genius]({song_url})")

                        youtube_url = extract_youtube_url(row.get('Media', ''))
                        if youtube_url:
                            video_id = youtube_url.split('watch?v=')[-1]
                            st.markdown(f"<iframe width='400' height='315' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' referrerpolicy='strict-origin-when-cross-origin' allowfullscreen></iframe>", unsafe_allow_html=True)

                        with st.expander("Show/Hide Lyrics"):
                            lyrics = str(row['Lyrics']).strip().replace('\n', '\n\n') if isinstance(row['Lyrics'], str) else "Lyrics not available."
                            st.markdown(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{lyrics}</pre>", unsafe_allow_html=True)
                        st.markdown("---")

if __name__ == '__main__':
    main()
