import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from functions import *



st.set_page_config(layout="wide")
st.title("⚽📝RFEF 24/25 Rulebook")

OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

col1,col2 = st.columns([1,2])


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What would be considered a goal if it was scored using a player´s hand?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted and OPENAI_API_KEY:
        response,relevant_chunks = get_response(text, OPENAI_API_KEY)
        st.info(response)
        print(relevant_chunks)
        processed_chunks = process_relevant_chunks(relevant_chunks)
        st.dataframe(processed_chunks)




