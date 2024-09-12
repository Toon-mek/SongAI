# Import necessary libraries
import pandas as pd
import streamlit as st
import gdown

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

# Add a sidebar for filtering songs by category (e.g., 'Year' or 'Artist')
st.sidebar.header('Filter Songs by Category')

# Assuming the dataset has columns 'Year' and 'Artist' to filter by
category_options = data_df['Year'].unique()  # Change 'Year' to the relevant column for filtering
selected_category = st.sidebar.selectbox('Select Year', options=category_options)

# Filter songs based on the selected category
filtered_songs = data_df[data_df['Year'] == selected_category]

# Display the filtered songs
st.write(f"### Songs from {selected_category}:")
st.write(filtered_songs[['Song Title', 'Artist', 'Year']])  # Display relevant columns
