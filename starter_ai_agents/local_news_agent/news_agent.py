import streamlit as st
from duckduckgo_search import DDGS
from datetime import datetime
import os
import openai # Use the official OpenAI library

# --- Configuration ---
# Set the OpenAI model to use
OPENAI_MODEL = "gpt-4o-mini" 

st.set_page_config(page_title="AI News Processor", page_icon="📰")
st.title("📰 News Inshorts Agent (OpenAI)")

# --- Sidebar for API Key Input ---
with st.sidebar:
    st.header("API Key Configuration")
    # Store the API key in the session state
    openai_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Optional: Display a status message
    if openai_key:
        st.success("API Key Entered! You can now process news.")
    else:
        st.warning("Please enter your OpenAI API key to run the agents.")

# --- Tool Function ---
def search_news(topic):
    """Search for news articles using DuckDuckGo"""
    with DDGS() as ddg:
        # Search for the topic limited to the current year and month for recency
        results = ddg.text(f"{topic} news {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            news_results = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return news_results
        return f"No news found for {topic}."

# --- Agent Instructions (System Prompts) ---

# We define the instructions for each step as simple strings
SEARCH_INSTRUCTIONS = """
You are a news search specialist. Your task is to:
1. Search for the most relevant and recent news on the given topic using the available tool.
2. Ensure the results are from reputable sources (this is handled by the search tool's output).
3. Return the raw search results in a structured format.
4. IMPORTANT: Do not generate any search result yourself. Use the search_news tool.
"""

SYNTHESIS_INSTRUCTIONS = """
You are a news synthesis expert. Your task is to:
1. Analyze the raw news articles provided.
2. Identify the key themes and important information.
3. Combine information from multiple sources.
4. Create a comprehensive but concise synthesis.
5. Focus on facts and maintain journalistic objectivity.
6. Write in a clear, professional style.
Provide a 2-3 paragraph synthesis of the main points.
"""

SUMMARY_INSTRUCTIONS = """
You are an expert news summarizer combining AP and Reuters style clarity with digital-age brevity.

Your task:
1. Core Information:
    - Lead with the most newsworthy development
    - Include key stakeholders and their actions
    - Add critical numbers/data if relevant
    - Explain why this matters now
    - Mention immediate implications

2. Style Guidelines:
    - Use strong, active verbs
    - Be specific, not general
    - Maintain journalistic objectivity
    - Make every word count
    - Explain technical terms if necessary

Format: Create a single paragraph of 250-400 words that informs and engages.
Pattern: [Major News] + [Key Details/Data] + [Why It Matters/What's Next]

Focus on answering: What happened? Why is it significant? What's the impact?

IMPORTANT: Provide ONLY the summary paragraph. Do not include any introductory phrases, 
labels, or meta-text like "Here's a summary" or "In AP/Reuters style."
Start directly with the news content.
"""

# --- Generic OpenAI Call Function (Replaces client.run) ---
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key):
    """Initializes and returns the OpenAI client."""
    return openai.OpenAI(api_key=api_key)

def run_agent_step(client, instructions, prompt, tool_defs=None, tool_choice="none"):
    """
    Executes a single step of the agent process using the OpenAI API.
    
    Args:
        client: The initialized openai.OpenAI client.
        instructions (str): The system prompt for the agent.
        prompt (str): The user's prompt/content.
        tool_defs (list, optional): List of tool definitions for function calling.
        tool_choice (str): Tool choice setting.
        
    Returns:
        str: The final text content from the model.
    """
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt}
    ]
    
    try:
        if tool_defs:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=tool_defs,
                tool_choice=tool_choice
            )
            # Handle function call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_news":
                    # Execute the local function
                    tool_output = search_news(function_args.get("topic"))
                    
                    # Send tool output back to the model
                    messages.append(response.choices[0].message)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output,
                    })
                    
                    # Second call to get the final text response
                    second_response = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                    )
                    return second_response.choices[0].message.content, tool_output
            
            # If no tool call was made but tool_defs were provided (unexpected for step 1)
            return response.choices[0].message.content, None
        
        # Standard chat completion (for synthesis and summary)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content, None
        
    except openai.APIError as e:
        st.error(f"OpenAI API Error: {e.message}")
        raise e
    except Exception as e:
        st.error(f"An unexpected error occurred during an API call: {e}")
        raise e


# --- News Processing Workflow (Updated) ---

# Define the tool structure for OpenAI's function calling API
import json
SEARCH_TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Searches for the most recent and relevant news articles on a given topic using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The news topic to search for, e.g., 'Tesla stock news'."
                    }
                },
                "required": ["topic"]
            },
        }
    }
]

def process_news(topic, api_key):
    """Run the news processing workflow using the OpenAI API."""
    
    # 1. Initialize client using the key from the sidebar
    openai_client = get_openai_client(api_key)

    with st.status("Processing news...", expanded=True) as status:
        # --- 1. Search Agent ---
        status.write("🔍 Searching for news...")
        # The search agent uses the function calling capability
        raw_news, tool_output = run_agent_step(
            client=openai_client,
            instructions=SEARCH_INSTRUCTIONS,
            prompt=f"Find recent news about {topic}",
            tool_defs=SEARCH_TOOL_DEFINITION,
            tool_choice={"type": "function", "function": {"name": "search_news"}}
        )
        
        # We need the raw news results (tool_output) for the next step
        # The model's *response* for this step is often just a confirmation, but we use the tool output.
        if not tool_output:
            status.update(label="Search Failed", state="error", expanded=False)
            st.error("The search agent failed to execute the news search tool.")
            return None, None, None
            
        raw_news_results = tool_output
        
        # --- 2. Synthesis Agent ---
        status.write("🔄 Synthesizing information...")
        synthesis_prompt = f"Synthesize these news articles:\n{raw_news_results}"
        synthesized_news, _ = run_agent_step(
            client=openai_client,
            instructions=SYNTHESIS_INSTRUCTIONS,
            prompt=synthesis_prompt
        )
        
        # --- 3. Summarize Agent ---
        status.write("📝 Creating summary...")
        summary_prompt = f"Summarize this synthesis:\n{synthesized_news}"
        final_summary, _ = run_agent_step(
            client=openai_client,
            instructions=SUMMARY_INSTRUCTIONS,
            prompt=summary_prompt
        )
        
        status.update(label="News Processing Complete!", state="complete", expanded=False)
        return raw_news_results, synthesized_news, final_summary

# --- User Interface ---
topic = st.text_input("Enter news topic:", value="artificial intelligence")
if st.button("Process News", type="primary"):
    if not topic.strip():
        st.error("Please enter a topic!")
    elif not openai_key:
        st.error("Please enter your OpenAI API Key in the sidebar to proceed.")
    else:
        try:
            # Pass the API key from the sidebar to the processing function
            raw_news, synthesized_news, final_summary = process_news(topic, openai_key)
            
            if final_summary:
                st.header(f"📝 News Summary: {topic}")
                st.markdown(final_summary)
                
                # Optional: Show detailed steps
                with st.expander("Show Detailed Steps"):
                    st.subheader("Raw News Articles")
                    st.code(raw_news, language='markdown')
                    st.subheader("Synthesized News")
                    st.markdown(synthesized_news)

        except Exception as e:
            # Error is already shown inside run_agent_step, but this catches broader issues
            if not st.session_state.get('error_shown'):
                 st.error(f"A workflow error occurred: {str(e)}")