from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Waking up your custom AI...")

# 1. Load the model you just trained from your local folder
model_path = "./my_news_summarizer"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def write_headline(article_text):
    # Tell the AI what its job is
    text = "summarize: " + article_text
    
    # Convert text to numbers
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
    
    # Generate the headline (num_beams=4 makes it think a bit harder for the best output)
    outputs = model.generate(inputs.input_ids, max_length=32, num_beams=4, early_stopping=True)
    
    # Convert numbers back to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("AI is ready!\n")
print("========================================")

# 2. Provide a sample news article to test! 
# (You can swap this text out for any article you find on the internet)
sample_article = """
SpaceX successfully launched its massive Starship rocket into orbit on Thursday morning from its Texas facility. 
The uncrewed test flight achieved several major milestones, including successful stage separation and reaching space. 
However, the company lost contact with the spacecraft during its reentry into Earth's atmosphere over the Indian Ocean.
Despite the loss of the ship, SpaceX engineers cheered the mission as a massive success for the future of space exploration.
"""

print("ORIGINAL ARTICLE:")
print(sample_article.strip())
print("\n----------------------------------------\n")

# 3. Ask the AI to write the headline
predicted_headline = write_headline(sample_article)

print("YOUR AI'S HEADLINE:")
print(predicted_headline)
print("========================================")