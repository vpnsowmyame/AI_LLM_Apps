import os
from uuid import uuid4
from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.log import logger
import streamlit as st

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Blog to Podcast Agent", page_icon="🎙️")
st.title("Blog to Podcast Agent")

# --- Sidebar: API Keys ---
st.sidebar.header("API Keys")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")

# Check if all keys are provided
keys_provided = all([openai_api_key, elevenlabs_api_key, firecrawl_api_key])

# --- Input: Blog URL ---
url = st.text_input("Enter the Blog URL:", "")

# --- Button: Generate Podcast ---
generate_button = st.button("Generate Podcast", disabled=not keys_provided)

if not keys_provided:
    st.warning("Please enter all required API keys to enable podcast generation.")

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a blog URL first.")
    else:
        # Set API keys as environment variables for Agno and Tools
        # Note: Agno tools often read keys directly from environment variables.
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key
        
        # Define the directory where ElevenLabsTools saves the audio
        save_dir = "audio_generations"
        os.makedirs(save_dir, exist_ok=True) # Ensure the directory exists locally

        with st.spinner("Processing... Scraping blog, summarizing and generating podcast 🎶"):
            try:
                # 1. Initialize the Agent
                blog_to_podcast_agent = Agent(
                    name="Blog to Podcast Agent",
                    id="blog_to_podcast_agent", # Corrected ID parameter
                    model=OpenAIChat(id="gpt-4o"),
                    tools=[
                        ElevenLabsTools(
                            voice_id="JBFqnCBsd6RMkjVDRZzb",
                            model_id="eleven_multilingual_v2",
                            target_directory=save_dir, # Tool will save the audio here
                        ),
                        FirecrawlTools(),
                    ],
                    description="You are an AI agent that can generate audio using the ElevenLabs API.",
                    instructions=[
                        "When the user provides a blog URL:",
                        "1. Use FirecrawlTools to scrape the blog content",
                        "2. Create a concise summary of the blog content that is NO MORE than 2000 characters long",
                        "3. The summary should capture the main points while being engaging and conversational",
                        "4. Use the ElevenLabsTools.generate_audio tool to convert the summary to audio",
                        "Ensure the summary is within the 2000 character limit to avoid ElevenLabs API limits",
                    ],
                    markdown=True,
                    debug_mode=True,
                )

                # 2. Run the Agent
                podcast: RunOutput = blog_to_podcast_agent.run(
                    f"Convert the blog content to a podcast: {url}"
                )

                # 3. Handle the Audio Output
                if podcast.audio and len(podcast.audio) > 0:
                    audio_object = podcast.audio[0]
                    
                    # When target_directory is set, the Audio object should have a filepath attribute
                    if hasattr(audio_object, 'filepath') and audio_object.filepath:
                        filepath = audio_object.filepath
                        
                        # Read the audio bytes from the file saved by the tool
                        with open(filepath, "rb") as f:
                             audio_bytes = f.read()

                        st.success(f"Podcast generated successfully! Saved to: {filepath} 🎧")
                        
                        # Display the audio player in Streamlit
                        st.audio(audio_bytes, format="audio/wav")

                        # Provide a download button
                        st.download_button(
                            label="Download Podcast",
                            data=audio_bytes,
                            file_name=os.path.basename(filepath),
                            mime="audio/wav"
                        )
                    else:
                        st.error("No file path found in the generated audio object. The tool may have failed to save the file.")
                else:
                    st.error("No audio was generated. Please check the logs (debug mode is enabled) for tool call errors.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Streamlit app error: {e}")
