import os
from dotenv import load_dotenv

def test_env_loading():
    """Test if environment variables are loaded correctly."""
    print("=== Testing Environment Loading ===")
    
    # Load .env file
    load_dotenv()
    print("✓ load_dotenv() called")
    
    # Test HUGGINGFACEHUB_API_TOKEN
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        print(f"✓ HUGGINGFACEHUB_API_TOKEN found: {hf_token[:10]}...{hf_token[-4:]}")
    else:
        print("✗ HUGGINGFACEHUB_API_TOKEN not found")
    
    # Test GROQ_API_KEY
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"✓ GROQ_API_KEY found: {groq_key[:10]}...{groq_key[-4:]}")
    else:
        print("✗ GROQ_API_KEY not found")
        
    # Test GOOGLE_API_KEY
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        print(f"✓ GOOGLE_API_KEY found: {gemini_key[:10]}...{gemini_key[-4:]}")
    else:
        print("✗ GOOGLE_API_KEY not found")
    
    return hf_token, groq_key, gemini_key

def test_hf_embeddings():
    """Test HuggingFace embeddings API."""
    print("\n=== Testing HuggingFace Embeddings API ===")
    
    try:
        from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
        
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("✗ No HUGGINGFACEHUB_API_TOKEN found")
            return False
        
        print("Creating HuggingFace embeddings client...")
        embeddings = HuggingFaceEndpointEmbeddings(
            model='BAAI/bge-m3',
            task="feature-extraction",
            huggingfacehub_api_token=hf_token
        )
        print("✓ Embeddings client created")
        
        # Test with a simple text
        test_text = "Hello, this is a test for Vietnamese financial news."
        print(f"Testing with text: '{test_text}'")
        
        result = embeddings.embed_query(test_text)
        print(f"✓ Embedding generated successfully! Dimension: {len(result)}")
        print(f"First 5 values: {result[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing HuggingFace embeddings: {e}")
        return False

def test_groq_api():
    """Test Groq API."""
    print("\n=== Testing Groq API ===")
    
    try:
        from langchain_groq import ChatGroq
        
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("✗ No GROQ_API_KEY found")
            return False
        
        print("Creating Groq chat client...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=100,
            api_key=groq_key,
        )
        print("✓ Groq client created")
        
        # Test with a simple query
        test_query = "Hello, how are you?"
        print(f"Testing with query: '{test_query}'")
        
        response = llm.invoke(test_query)
        print(f"✓ Groq response received: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Groq API: {e}")
        return False
    
def test_gemini_api():
    """Test Gemini API."""
    print("\n=== Testing Gemini API ===")
    
    try:
        from langchain_google_genai import GoogleGenerativeAI
        
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            print("✗ No GOOGLE_API_KEY found")
            return False
        
        print("Creating Gemini chat client...")
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=100,
            api_key=gemini_key,
        )
        print("✓ Gemini client created")
        
        # Test with a simple query
        test_query = "Hello, how are you?"
        print(f"Testing with query: '{test_query}'")
        
        response = llm.invoke(test_query)
        print(f"✓ Gemini response received: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Gemini API: {e}")
        return False

def main():
    """Main test function."""
    print("HuggingFace &  API Token Test")
    print("=" * 50)
    
    # Test environment loading
    hf_token, groq_key, gemini_key = test_env_loading()
    
    # Test HuggingFace embeddings if token available
    if hf_token:
        test_hf_embeddings()
    else:
        print("\n⚠️  Skipping HuggingFace test - no token found")
    
    # Test Groq API if key available
    if groq_key:
        test_groq_api()
    else:
        print("\n⚠️  Skipping Groq test - no key found")
    
    # Test Gemini API if key available
    if gemini_key:
        test_gemini_api()
    else:
        print("\n⚠️  Skipping Gemini test - no key found")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
