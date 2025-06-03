import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from functions import *
import tempfile
import shutil 
from pathlib import Path


# Setup
st.set_page_config(layout="wide")
st.title("⚽📝RFEF & NBA 24/25 Rulebook")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    # Input for OpenAI API Key
    api_key_input = st.text_input("OpenAI API Key", type="password")
    if api_key_input:
        OPENAI_API_KEY = api_key_input
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and OPENAI_API_KEY:
        if st.button("Upload to DB"):
            try:
                # Create a temporary directory
                temp_dir = Path("temp_upload")
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded file using its original name
                temp_file_path = temp_dir / uploaded_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Use the function with actual file path
                upload_file_in_db(str(temp_file_path), OPENAI_API_KEY)

                # Clean up the temporary folder after processing
                shutil.rmtree(temp_dir)

                st.success("File successfully uploaded to the RAG database.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Text input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a rule-related question:")
        submitted = st.form_submit_button("Send")
        if submitted and user_input and OPENAI_API_KEY:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get response
            response, relevant_chunks = get_response(user_input, OPENAI_API_KEY)
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    # Display chat history
    st.subheader("💬 Chat History")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.success(f"🧍 You: {msg['content']}")
        else:
            st.info(f"🤖 Assistant: {msg['content']}")

    # Optionally show relevant chunks as a table for the last answer
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        processed_chunks = process_relevant_chunks(relevant_chunks)
        st.dataframe(processed_chunks)
