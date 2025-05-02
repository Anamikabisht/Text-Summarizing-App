import streamlit as st
from transformers import pipeline
import pandas as pd
import nltk

# Download sentence tokenizer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load summarizer pipeline
summarizer = pipeline("summarization")

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📚 Text Summarizer App")
st.write("Paste your full text below. Each sentence will be summarized individually.")

# Text input box
input_text = st.text_area("Enter your text here", height=300)

# Button to trigger summarization
if st.button("Summarize Sentences"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Split input into sentences
        sentences = sent_tokenize(input_text.strip())
        
        # Generate summaries
        summaries = []
        for s in sentences:
            try:
                result = summarizer(s, max_length=50, min_length=10, do_sample=False)
                summaries.append(result[0]['summary_text'])
            except:
                summaries.append("❗ Unable to summarize (too short or invalid sentence)")

        # Display in a DataFrame
        df = pd.DataFrame({
            "Sentence": sentences,
            "Summary": summaries
        })
        st.dataframe(df, use_container_width=True)
