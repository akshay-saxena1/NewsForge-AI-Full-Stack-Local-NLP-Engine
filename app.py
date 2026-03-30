import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Headline Generator", page_icon="📰")

# --- MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    # Update this path to your local folder
    model_path = "./my_news_summarizer"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# --- CORE LOGIC ---
def write_headline(article_text):
    text = "summarize: " + article_text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids, 
        max_length=32, 
        num_beams=4, 
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- USER INTERFACE ---
st.title("📰 AI Headline Generator")
st.markdown("Paste your article below, and let the custom AI craft a headline for you.")

# Input area
article_input = st.text_area("Article Text:", placeholder="Paste your paragraph here...", height=300)

# Execution button
if st.button("Generate Headline"):
    if article_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        with st.spinner("AI is thinking..."):
            try:
                headline = write_headline(article_input)
                
                # Display result
                st.subheader("Generated Headline:")
                st.success(f"**{headline.upper()}**")
                
                # Optional: Add a 'Copy' feature (built-in to st.code)
                st.code(headline.upper())
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.divider()
st.caption("Powered by T5 and Streamlit")