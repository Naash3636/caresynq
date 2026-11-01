import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API KEY NOT FOUND")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply: CARESYNQ Gemini connection successful"
        )

        print("SUCCESS!")
        print(response.text)

    except Exception as e:
        print("GEMINI ERROR:")
        print(e)