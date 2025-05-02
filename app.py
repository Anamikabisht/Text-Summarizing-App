
import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(page_title="Line-wise Text Summarizer", layout="wide")
st.title("📄 Line-wise Text Summarizer")

st.markdown("Paste your full text below. Each line will be summarized individually.")

# Text input
text = st.text_area("Enter multiline text here", height=300)

# Summarizer pipeline
@st.cache_resource
def get_summarizer():
    return pipeline("summarization")

if st.button("Summarize Lines"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        summarizer = get_summarizer()
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        sent_score = {}

        for i, line in enumerate(lines):
            try:
                summary = summarizer(line, max_length=50, min_length=5, do_sample=False)[0]['summary_text']
                sent_score[f"Line {i+1}"] = summary
            except Exception as e:
                sent_score[f"Line {i+1}"] = f"[Error summarizing line] {e}"

        # Display in table
        df = pd.DataFrame(list(sent_score.items()), columns=["Line", "Summary"])
        st.dataframe(df)
