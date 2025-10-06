import streamlit as st
import json
import requests
import time

# --- Configuration ---
# The model used for generating content
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# The agent's instructions, defining its persona and output format.
# Note: Since the standard OpenAI API call doesn't natively include real-time grounding,
# the prompt relies on the model's up-to-date knowledge and its instruction to act as an analyst.
SYSTEM_PROMPT = (
    "Act as a world-class financial analyst and investment advisor. "
    "Always use markdown tables to display financial/numerical data for clarity and easy comparison. "
    "For text data (like qualitative analysis or recommendations), use bullet points and small paragraphs. "
    "Provide a detailed and well-structured response."
)

# --- Helper Functions ---

def exponential_backoff_fetch(url, payload, api_key, max_retries=5):
    """
    Handles API fetching with exponential backoff for transient errors.
    Uses the API key in the Authorization header for OpenAI.
    """
    if not api_key:
        st.error("API Key is missing.")
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    for attempt in range(max_retries):
        try:
            # Full URL is used directly without appending the key
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Successful response
            return response.json()

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors that might require retrying (e.g., 429 Rate Limit)
            if response.status_code == 429 or response.status_code >= 500:
                delay = 2 ** attempt
                st.warning(f"Rate limit or server error ({response.status_code}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Other errors (e.g., 401 Unauthorized, 400 Bad Request) are critical
                st.error(f"OpenAI API Request Failed: {e}. Error details: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: {e}")
            return None
            
    st.error("Max retries exceeded. The API request failed.")
    return None


def get_financial_analysis(query, api_key):
    """
    Calls the OpenAI Chat Completions API with system instructions.
    """
    st.info(f"Analyzing query: '{query}' using {OPENAI_MODEL_NAME}...")
    
    # Payload structure for the OpenAI Chat Completions API
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    }

    # Use the helper function to call the API with backoff
    result = exponential_backoff_fetch(OPENAI_API_ENDPOINT, payload, api_key)

    if not result:
        st.error("Could not retrieve a valid response from the OpenAI API.")
        return None

    try:
        # Extract the generated text from OpenAI's standard response structure
        generated_text = result.get('choices', [{}])[0].get('message', {}).get('content', 'No analysis generated.')
        
        # OpenAI responses do not contain the same grounding metadata structure, so sources are omitted.
        return generated_text

    except Exception as e:
        st.error(f"Error processing API response structure: {e}")
        return None


# --- Streamlit Application Layout ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="OpenAI Finance Agent", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    st.title("💰 AI Financial Analyst (Powered by OpenAI)")
    st.markdown("---")

    # Sidebar for API Key Input and Information
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input(
            "Enter OpenAI API Key", 
            type="password", 
            placeholder="sk-..."
        )
        
        st.markdown(
            "This agent uses the **OpenAI GPT-4o** model to analyze financial questions. "
            "It is instructed to format all numerical data into clear tables."
        )
        st.markdown("---")
        st.code(f"Model: {OPENAI_MODEL_NAME}", language="text")

    # Main interaction area
    
    user_query = st.text_area(
        "Enter your financial question or stock analysis request (e.g., 'Analyze the Q3 earnings for Tesla (TSLA) and provide a valuation summary.'):",
        height=100
    )

    if st.button("Get Analysis", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar to proceed.")
            return
        if not user_query:
            st.warning("Please enter a financial question before clicking 'Get Analysis'.")
            return

        with st.spinner("The AI Finance Agent is gathering data and preparing the analysis..."):
            
            # --- API Call Logic ---
            analysis_text = get_financial_analysis(user_query, api_key)
            
            # --- Display Results ---
            if analysis_text:
                st.subheader("📊 Financial Analysis")
                st.markdown(analysis_text)
                st.success("Analysis Complete!")
    
    st.markdown("---")
    st.caption("Disclaimer: This tool provides AI-generated financial insights and should not be considered professional investment advice.")


if __name__ == "__main__":
    main()
