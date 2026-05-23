import torch
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ignore harmless warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="AI News Summarizer Hub",
    description="A premium full-stack local AI Headline & Summary Generator"
)

# --- MODEL LOADING & INITIALIZATION ---
print("Waking up your custom AI model...")
model_path = "./my_news_summarizer"

try:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Determine the fastest available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model onto device: {device.upper()}")
    model.to(device)
    print("AI Model loaded successfully and is ready!")
except Exception as e:
    print(f"Error loading local model from {model_path}: {e}")
    print("Falling back to loading 't5-small' model as safety net...")
    try:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print("Fallback model 't5-small' loaded successfully!")
    except Exception as fallback_error:
        print(f"Critical error loading fallback model: {fallback_error}")
        raise fallback_error

# --- CORE LOGIC ---
def generate_summary_or_headline(article_text: str) -> str:
    text = "summarize: " + article_text
    # Convert text to tokens and move to selected device
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # Generate summary tokens
    outputs = model.generate(
        inputs.input_ids, 
        max_length=32, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode back to clean text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- API TYPES ---
class ArticleRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---

# 1. SERVE THE NEW FRONTEND UI
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # This reads the new file we just created instead of storing HTML in Python
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# 2. HANDLE THE BUTTON CLICK FROM THE UI
@app.post("/generate")
async def generate_endpoint(payload: ArticleRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty.")
    
    try:
        # Pass the UI text directly into your perfectly working T5 logic
        summary_result = generate_summary_or_headline(payload.text)
        
        # Send it back to the React UI
        return {"headline": summary_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Run server when executing file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)