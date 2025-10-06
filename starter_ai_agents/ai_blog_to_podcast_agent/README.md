## Blog to Podcast Agent

This is a Streamlit-based application that allows users to convert any blog post into a podcast. The app uses OpenAI's GPT-4 model for summarization, Firecrawl for scraping blog content, and ElevenLabs API for generating audio. Users simply input a blog URL, and the app will generate a podcast episode based on the blog.

## Features

- **Blog Scraping**: Scrapes the full content of any public blog URL using Firecrawl API.

- **Summary Generation**: Creates an engaging and concise summary of the blog (within 2000 characters) using OpenAI GPT-4.

- **Podcast Generation**: Converts the summary into an audio podcast using the ElevenLabs voice API.

- **API Key Integration**: Requires OpenAI, Firecrawl, and ElevenLabs API keys to function, entered securely via the sidebar.

## In the app interface:
    - Enter your OpenAI, ElevenLabs, and Firecrawl API keys in the sidebar.

    - Input the blog URL you want to convert.

    - Click "🎙️ Generate Podcast".

    - Listen to the generated podcast or download it.
