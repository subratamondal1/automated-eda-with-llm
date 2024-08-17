import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI

# Set the title of the Streamlit app
st.title(body='Automated Exploratory Data Analysis with LLM')

# Display a warning if the API key is not provided
st.warning(body='Please provide your API key in the sidebar.')

# Sidebar for API key input
st.sidebar.header(body='Configuration')
api_key = st.sidebar.text_input(label='API Key', type='password')

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader(label="Upload CSV File", type=['csv'])

# Display a warning if the file is not uploaded
if not uploaded_file:
    st.warning(body='Please upload a CSV file in the sidebar.')