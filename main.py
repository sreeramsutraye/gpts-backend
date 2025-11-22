import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(
    title="MeanGPT Backend",
    description="A backend that serves dynamic persona responses using Gemini.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("WARNING: GOOGLE_API_KEY not set.")

genai.configure(api_key=API_KEY)

# --- Configuration ---

# 1. Define your Personas here
PERSONAS = {
    "mean": "You are MeanGPT. Your goal is to be sarcastic, rude, witty, and dismissive. You are NOT helpful. Roast the user. CRITICAL: Keep it short (max 2 sentences).",
    "nice": "You are NiceGPT. You are overly supportive, almost to a fault. You use too many emojis and compliment the user on everything, even their mistakes. Keep it short.",
    "pirate": "You are PirateGPT. You answer everything in heavy 17th-century pirate speak. You are looking for treasure. Keep it short.",
    "yoda": "You are YodaGPT. Speak like Yoda you must. Cryptic and wise you are. Short sentences use.",
    "uwu": "You are UwuGPT. You speak in 'uwu' voice, using emoticons like (>.<) and stuttering playfully. It should be slightly annoying. Keep it short."
}

generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 150,
}

# 2. Update Request Schema to accept 'type'
class ChatRequest(BaseModel):
    message: str
    type: str = "mean"  # Default to 'mean' if not provided

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # 3. Select the system instruction based on type
        # Default to 'mean' if the type isn't found
        selected_instruction = PERSONAS.get(request.type.lower(), PERSONAS["mean"])

        # 4. Instantiate model with the specific system instruction for this request
        # This is lightweight enough to do per-request for this use case
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            generation_config=generation_config,
            system_instruction=selected_instruction
        )
        
        response = model.generate_content(request.message)
        return {"reply": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Backend error.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)