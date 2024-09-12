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

# Display the original dataset in Streamlit
st.write("Original Dataset:")
st.write(data_df.head())

# Add a sidebar for selecting a category and filtering songs
st.sidebar.header('Filter Songs by Category')

# Allow user to select which column to filter by
filter_column = st.sidebar.selectbox('Select a Category to Filter By', options=['Artist', 'Album', 'Release Date', 'Song Title'])

# Get unique values from the selected column to create filter options
category_options = data_df[filter_column].dropna().unique()  # Drop any NaN values and get unique entries
selected_category = st.sidebar.selectbox(f'Select {filter_column}', options=category_options)

# Filter songs based on the selected category
filtered_songs = data_df[data_df[filter_column] == selected_category]

# Display the filtered songs
st.write(f"### Songs Filtered by {filter_column}: {selected_category}")
st.write(filtered_songs[['Song Title', 'Artist', 'Album', 'Release Date']])  # Display relevant columns
