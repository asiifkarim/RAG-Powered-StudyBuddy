import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_connection():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return False
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # List available models
        print("Available models:")
        for model in genai.list_models():
            if "gemini" in model.name:
                print(f"- {model.name}")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')  # Use the latest stable version

        # Test generation
        response = model.generate_content("What is 2+2?")
        
        if response and response.text:
            print("\n✅ Gemini API connection successful")
            print(f"Test response: {response.text}")
            return True
        else:
            print("\n❌ Error: No response generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Error connecting to Gemini API: {str(e)}")
        print("\nPlease check:")
        print("1. Your API key is valid")
        print("2. You have access to Gemini Pro")
        print("3. Your API key has billing enabled")
        return False

if __name__ == "__main__":
    test_gemini_connection()