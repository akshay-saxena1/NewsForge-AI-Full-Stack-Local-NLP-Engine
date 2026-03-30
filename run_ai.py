import warnings
# This hides some of the harmless background warnings to keep your terminal looking clean
warnings.filterwarnings("ignore") 

from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Waking up your custom AI... (This takes a few seconds)")

# 1. Load the model and tokenizer from your local folder
model_path = "./my_news_summarizer"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# 2. Define the function that actually generates the headline
def write_headline(article_text):
    # Tell the T5 model what its job is
    text = "summarize: " + article_text
    
    # Convert text to numbers
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the headline 
    # num_beams=4 makes the AI explore a few different options before picking the best one
    outputs = model.generate(inputs.input_ids, max_length=32, num_beams=4, early_stopping=True)
    
    # Convert the output numbers back into readable text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 3. The Interactive Loop
print("\n=========================================")
print("     YOUR AI HEADLINE GENERATOR IS READY   ")
print("=========================================")

while True:
    print("\nPaste your article or paragraph below.")
    print("(Press 'Enter' TWICE to generate the headline, or type 'QUIT' to exit)")
    print("----------------------------------------------------------------------")
    
    lines = []
    while True:
        line = input()
        
        # Check if the user wants to quit
        if line.strip().upper() == "QUIT":
            print("\nShutting down AI. Goodbye!")
            exit()
            
        # If the user hits Enter on a completely blank line, stop reading
        if line == "":
            break
            
        lines.append(line)

    # Join all the typed lines into one single block of text
    paragraph = "\n".join(lines).strip()
    
    # If they just pressed Enter without typing anything, ask again
    if not paragraph:
        continue

    # Generate and print the result!
    print("\n[AI is thinking...]")
    predicted_headline = write_headline(paragraph)
    
    print("\n📰 YOUR HEADLINE:")
    print(f">>> {predicted_headline.upper()} <<<")
    print("=========================================\n")