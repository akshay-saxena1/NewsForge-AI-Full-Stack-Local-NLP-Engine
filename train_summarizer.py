import pandas as pd
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split

print("1. Loading and cleaning the dataset...")

# Load the dataset (make sure this CSV is in the same folder as this script)
df = pd.read_csv('news_summary_more.csv', encoding='latin-1')

# Keep only the columns we need and drop empty rows
df = df[['text', 'headlines']].dropna()

# We will take a subset of 5,000 articles to keep training fast for your RTX 4050.
df = df.sample(n=5000, random_state=42)

# T5 Model requires the word "summarize: " before the input text to know what task to do
df['text'] = "summarize: " + df['text']

# Split into training (80%) and testing (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert Pandas DataFrames into Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print("2. Downloading the T5-Small AI Model & Tokenizer...")
# Tokenizers convert words into numbers that the AI can understand
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("3. Formatting the data for the model...")
# This function converts our text and headlines into number tokens
def preprocess_function(examples):
    # Tokenize the input text (the article)
    inputs = tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length")
    # Tokenize the target text (the headline)
    targets = tokenizer(examples["headlines"], max_length=32, truncation=True, padding="max_length")
    
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply the tokenization to our datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

print("4. Setting up the Trainer for the RTX 4050 GPU...")
# Here we define the training rules (10 epochs, specific batch sizes for 6GB VRAM)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # UPDATED
    learning_rate=2e-5,
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2, 
    num_train_epochs=10, 
    predict_with_generate=True,
    fp16=True, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer, # UPDATED: Changed from tokenizer=tokenizer
)

print("5. BEGINNING TRAINING (This will take some time)...")
trainer.train()

print("6. Training Complete! Saving the finely-tuned model...")
# Save the model to a folder so you can use it later!
trainer.save_model("./my_news_summarizer")
tokenizer.save_pretrained("./my_news_summarizer")

print("All done! Your custom AI is saved in the 'my_news_summarizer' folder.")