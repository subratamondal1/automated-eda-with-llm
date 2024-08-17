import os
import time
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set the title of the Streamlit app
st.title('Automated Exploratory Data Analysis with LLM')

# Sidebar for API key input
st.sidebar.header('Configuration')
api_key = st.sidebar.text_input('API Key', type='password')

if not api_key:
    st.warning("Provide your GPT4o API Key")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])

if not uploaded_file:
    st.warning("Please upload your csv file.")

if api_key and uploaded_file:
    if st.button('Analyze'):
        with st.spinner('Analyzing... Please wait.'):
            try:
                start_time = time.time()

                # Load the data
                data = pd.read_csv(uploaded_file)

                # Initialize the chat model with the provided API key
                llm = ChatOpenAI(api_key=api_key, model="gpt-4o")

                # Create an agent that interacts with the DataFrame
                agent = create_pandas_dataframe_agent(
                    llm, data, allow_dangerous_code=True
                )

                # Define EDA tasks
                eda_tasks = [
                    "Inspect the dataset for any columns that may be considered meaningless. For each column identified, provide a justification for why it is deemed unnecessary, remove it from the dataset, and update the data accordingly.",
                    "Examine each column to determine its data type. Classify columns as categorical, nominal, ordinal, or numerical, and further specify if they are discrete or continuous. Explain the reasoning behind each classification.",
                    "Conduct data cleaning by addressing missing values. Describe the techniques you employ for handling these values and justify your choices. Summarize the modifications made to the dataset.",
                    "Identify numerical columns and compute descriptive statistics. Analyze these statistics to extract meaningful insights and articulate your interpretations.",
                    "Investigate relationships between variables through correlation analysis. Provide insights into any significant correlations discovered and discuss their implications.",
                    "Detect any patterns, trends, or anomalies within the data. Analyze these findings and explain their significance and potential impact.",
                    "Based on your analysis, compile a comprehensive summary of insights and conclusions. Highlight the key findings and their broader implications."
                ]

                # Execute EDA tasks
                st.markdown("# Exploratory Data Analysis (EDA) Results")
                for i, task in enumerate(eda_tasks, 1):
                    try:
                        st.markdown("---")

                        # Prepare a markdown prompt for the LLM
                        markdown_prompt = f"**{task}**\n\n_Convert the output into Markdown_"
                        response = agent.invoke(markdown_prompt)
                        output = response.get('output', '')

                        # Display the markdown output
                        st.markdown(output, unsafe_allow_html=True)

                        # Update the data if necessary (e.g., after removing columns or cleaning data)
                        # This requires the agent to return the updated DataFrame
                        if 'updated_data' in response:
                            data = response['updated_data']

                    except Exception as e:
                        st.error(f"An error occurred while processing the task: {task}. Error: {e}")

                end_time = time.time()
                total_time = end_time - start_time
                st.success(f"Analysis completed in **{total_time:.2f} seconds**.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info('_Please provide both an API key and upload a CSV file to enable the Analyze button._')