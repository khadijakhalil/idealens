import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set. Check your .env file.")
    exit(1)

# Initialize Gemini Client
# The SDK automatically uses GEMINI_API_KEY, but passing it explicitly guarantees clarity.
client = genai.Client(api_key=api_key)

try:
    # Using gemini-2.5-flash as gemini-1.5-pro is not available and gemini-2.5-pro has free tier quota limitations
    print("Sending request to gemini-2.5-flash...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Say: IdeaLens is ready',
    )
    
    print("\nAPI Response:")
    print(response.text)
except Exception as e:
    print(f"\nAn error occurred: {e}")



