import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

app = FastAPI(
    title="MeanGPT Backend",
    description="A backend that serves dynamic persona responses using Gemini.",
    version="1.2.0",
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
    print("\nCRITICAL WARNING: GOOGLE_API_KEY is missing.")
else:
    genai.configure(api_key=API_KEY)

# --- Configuration ---

PERSONAS = {
    "mean": "You are MeanGPT. Your goal is to be sarcastic, rude, witty, and dismissive. You are NOT helpful. Roast the user. CRITICAL: Keep it short (max 2 sentences).",
    "nice": "You are NiceGPT. You are overly supportive, almost to a fault. You use too many emojis and compliment the user on everything, even their mistakes. Keep it short.",
    "pirate": "You are PirateGPT. You answer everything in heavy 17th-century pirate speak. You are looking for treasure. Keep it short.",
    "yoda": "You are YodaGPT. Speak like Yoda you must. Cryptic and wise you are. Short sentences use.",
    "uwu": "You are UwuGPT. You speak in 'uwu' voice, using emoticons like (>.<) and stuttering playfully. It should be slightly annoying. Keep it short.",
}

generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 150,
}

# 2. Adjusted Safety Settings
# This allows the model to be "rude" without getting blocked immediately.
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class ChatRequest(BaseModel):
    message: str
    type: str = "nice"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if not API_KEY:
            raise HTTPException(
                status_code=500, detail="Server Configuration Error: API Key missing."
            )

        if not request.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        selected_instruction = PERSONAS.get(request.type.lower(), PERSONAS["nice"])

        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config=generation_config,
                safety_settings=safety_settings,  # Apply settings here
                system_instruction=selected_instruction,
            )
        except TypeError as e:
            print(f"Library Version Error: {e}")
            raise HTTPException(status_code=500, detail="Server library outdated.")

        response = model.generate_content(request.message)

        # 3. Safer Response Handling
        # Instead of crashing on response.text, we check if candidates exist.
        if response.candidates and response.candidates[0].content.parts:
            return {"reply": response.text}
        else:
            # If safety filters blocked it completely:
            print(f"Blocked. Finish Reason: {response.prompt_feedback}")
            return {
                "reply": "I refused to answer that. (My safety filters triggered because I was too mean)."
            }

    except Exception as e:
        print(f"Backend Error: {e}")
        # Return a generic error to the frontend so it doesn't hang
        return {"reply": "My brain broke. Try again."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
