import streamlit as st
import os
import asyncio
from datetime import datetime, timedelta
import re
import csv
from time import sleep
from urllib.parse import urljoin
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
import requests

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from typing import AsyncGenerator
from typing_extensions import override

# --- Hide Sidebar ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# --- Utility functions ---
headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def parse_time_range(time_range):
    unit = time_range[-1]
    value = int(time_range[:-1])
    if unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError("Invalid time range format.")

def parse_posted_time(posted_str):
    now = datetime.now()
    match = re.search(r'(\d+)\s+(minute|hour|day)s?\s+ago', posted_str)
    if match:
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'minute':
            return now - timedelta(minutes=value)
        elif unit == 'hour':
            return now - timedelta(hours=value)
        elif unit == 'day':
            return now - timedelta(days=value)
    return now

def get_article(card):
    try:
        headline = card.find('h4', 's-title').text
        source = card.find("span", 's-source').text
        posted = card.find('span', 's-time').text.replace('·', '').strip()
        description = card.find('p', 's-desc').text.strip()
        raw_link = card.find('a').get('href')
        unquoted_link = requests.utils.unquote(raw_link)
        pattern = re.compile(r'RU=(.+)\/RK')
        match = re.search(pattern, unquoted_link)
        clean_link = match.group(1) if match else unquoted_link
        posted_time = parse_posted_time(posted)
        return (headline, source, posted, description, clean_link, posted_time)
    except Exception as e:
        print(f"Skipping article due to error: {e}")
        return None

def get_the_news(search, time_range='1h'):
    url = f'https://news.search.yahoo.com/search?p={requests.utils.quote(search)}'
    cutoff = datetime.now() - parse_time_range(time_range)
    articles = []
    links = set()
    pages = 0
    while pages < 5:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        cards = soup.find_all('div', 'NewsArticle')
        found = False
        for card in cards:
            article = get_article(card)
            if article and article[-1] >= cutoff:
                found = True
                if article[4] not in links:
                    links.add(article[4])
                    articles.append(article[:-1])  # Exclude datetime from output
        if not found:
            break
        nxt = soup.find('a', 'next')
        if not nxt or not nxt.get('href'):
            break
        url = urljoin('https://news.search.yahoo.com', nxt['href'])
        pages += 1
        sleep(1)
    return articles

# --- NewsPostAgent ---
class NewsPostAgent(BaseAgent):
    news_agent: LlmAgent
    filter_agent: LlmAgent
    post_agent: LlmAgent
    sequential: SequentialAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, model):
        news_agent = LlmAgent(
            name="NewsGenerator",
            model=model,
            instruction="""
                Summarize the entire 'news' input into a coherent, well-structured 250-word paragraph. 
                Ensure the summary captures the key points, important facts, names, locations, and any relevant timestamps. 
                Use professional journalistic tone. Do not include any personal opinion or fictional additions. 
                Aim for factual clarity, coherence, and relevance across all segments of the given news.
                """,
            input_schema=None,
            output_key="draft",
        )

        filter_agent = LlmAgent(
            name="ContentFilter",
            model=model,
            instruction="""
                You are a strict content safety filter.
                Carefully analyze the input text (the 'draft') and determine if it is completely safe and appropriate for public dissemination on professional platforms like LinkedIn or Twitter.

                Your rules are:

                - Reject and mark as 'Unsafe' any text containing:
                - Hate speech, slurs, discriminatory or prejudiced language against any group or individual.
                - Political bias, propaganda, or statements that could inflame partisan conflicts.
                - Graphic descriptions or glorification of violence, war, or disturbing imagery.
                - Personal attacks, defamation, or language that targets individuals or groups.
                - Content that promotes illegal activities, self-harm, or dangerous behavior.
                - Sensationalism, exaggeration, or emotionally manipulative wording.
                - Use of emojis, hashtags, slogans, or informal internet slang that undermine professionalism.
                
                - Only respond with exactly one of the following, nothing else, no explanation or elaboration:
                - 'Safe' — if the content is strictly factual, neutral, professional, non-inflammatory, and safe.
                - 'Unsafe' — if the content violates any of the above rules, or if there is any doubt about its safety.

                Be uncompromising and conservative. If you are unsure whether content is safe, respond with 'Unsafe'.

                DO NOT explain your decision. DO NOT add any commentary. Respond ONLY with 'Safe' or 'Unsafe'.
                """,
            input_schema=None,
            output_key="safety",
        )


        post_agent = LlmAgent(
            name="PostFormatter",
            model=model,
            instruction="""
                If 'safety' is 'Safe', take the 'draft' and format it as a polished, engaging social media post suitable for platforms like LinkedIn or Twitter. 
                Keep it professional and informative. Begin with a compelling hook. Break long text into readable parts if necessary. 
                Have hashtags and emojis. Output only the final post.
                If 'safety' is not 'Safe', do not output anything.
                """,
            input_schema=None,
            output_key="final_post",
        )

        sequential = SequentialAgent(
            name="NewsPipeline",
            sub_agents=[news_agent, filter_agent, post_agent]
        )

        super().__init__(
            name="NewsPostAgent",
            news_agent=news_agent,
            filter_agent=filter_agent,
            post_agent=post_agent,
            sequential=sequential,
            sub_agents=[sequential]
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async for event in self.sequential.run_async(ctx):
            yield event

async def run_news_agent(model, combined_news):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="news_app",
        user_id="u1",
        session_id="sess1",
        state={"news": combined_news}
    )
    agent = NewsPostAgent(model)
    runner = Runner(agent=agent, app_name="news_app", session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text=combined_news)])
    events = runner.run(user_id="u1", session_id="sess1", new_message=content)
    for _ in events:
        pass
    session = await session_service.get_session(app_name="news_app", user_id="u1", session_id="sess1")
    return session.state.get("safety", ""), session.state.get("final_post", ""), session.state

def main():
    # --- Model Options ---
    GOOGLE_API_MODELS = ["gemini-2.0-flash", "gemini-2.0-pro"]
    GOOGLE_VERTEX_MODELS = ["gemini-1.5-pro-001", "gemini-1.5-flash-001"]
    OPENAI_MODELS = ["openai/gpt-4.1", "openai/gpt-4o"]
    ANTHROPIC_MODELS = ["anthropic/claude-sonnet-4-20250514"]
    ALL_MODELS = {
        "Google API": GOOGLE_API_MODELS,
        "Google Vertex API": GOOGLE_VERTEX_MODELS,
        "OpenAI": OPENAI_MODELS,
        "Anthropic": ANTHROPIC_MODELS
    }

    # --- Navigation Logic ---
    st.set_page_config(layout="wide")
    if "page" not in st.session_state:
        st.session_state["page"] = "llm_connector"

    if st.session_state["page"] == "llm_connector":
        st.title("🗞️ AI-powered News Agent")
        st.subheader("🔌 LLM Connector")
        llm_provider = st.selectbox("Select LLM Provider:", list(ALL_MODELS.keys()))
        selected_model = st.selectbox("Select Model:", ALL_MODELS[llm_provider])
        st.session_state["llm_model"] = selected_model
        st.session_state["provider"] = llm_provider

        if llm_provider == "Google API":
            api_key = None
            os.environ["GOOGLE_API_KEY"] = ""
            os.environ["OPENAI_API_KEY"] = ""
            os.environ["ANTHROPIC_API_KEY"] = ""
            api_key = st.text_input("Enter your Google API Key:", type="password")
            if api_key:
                os.environ.clear()
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
                st.success("Google API key set.")

        elif llm_provider == "Google Vertex API":
            api_key = None
            os.environ["GOOGLE_API_KEY"] = ""
            api_key = st.text_input("Enter your Google Vertex AI Key:", type="password")
            if api_key:
                os.environ.clear()
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
                st.success("Google Vertex AI key set.")

        elif llm_provider == "OpenAI":
            api_key = None
            os.environ["OPENAI_API_KEY"] = ""
            api_key = st.text_input("Enter your OpenAI API Key:", type="password")
            if api_key:
                os.environ.clear()
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("OpenAI API key set.")

        elif llm_provider == "Anthropic":
            api_key = None
            os.environ["ANTHROPIC_API_KEY"] = ""
            api_key = st.text_input("Enter your Anthropic API Key:", type="password")
            if api_key:
                os.environ.clear()
                os.environ["ANTHROPIC_API_KEY"] = api_key
                st.success("Anthropic API key set.")

        if api_key: 
            if st.button("▶️ Start App"):
                st.session_state["page"] = "news_poster"
                st.rerun()
        else:
            st.warning("API key not set. Please enter a valid API key and press enter to show Start App Button.")


        st.info("About this App: This application is Developed and built by Sanjeev Mohan(mysanjeev99@gmail.com), Balaharidass Ramachandran and Karthikeyan S. \n" \
        "Social Media Agent - is an intelligent content creation tool designed to generate social media posts based on a given topic and timeframe. " \
        "It features built-in content filtering to ensure quality and relevance—if the generated content doesn't align with the intended topic or standards, " \
        "it won't be published. \n\n" \
        "This application uses a multi-agent architecture to ensure the news content it generates is informative, safe, and ready for public dissemination. " \
        "At its core, it leverages a sequential pipeline of three specialized LLM agents that work together to process and refine the input. " \
        "The first agent, called the NewsGenerator, takes a collection of recent news articles with 'get_news' Tool and produces a single, coherent summary in a formal and journalistic tone. " \
        "This summary captures the essential facts, events, and context from all the articles combined. Once the summary is generated, it is passed to the second agent, " \
        "the ContentFilter. This agent performs a strict safety evaluation, determining whether the text is suitable for public sharing on professional platforms like LinkedIn or Twitter. " \
        "It checks for content that might be considered hateful, politically biased, graphic, or otherwise inappropriate. Based on this assessment, " \
        "it returns either 'Safe' or 'Unsafe' with no additional commentary. If the content is deemed safe, it proceeds to the third and final agent, " \
        "the PostFormatter. This agent transforms the formal summary into an engaging, platform-appropriate social media post. " \
        "It restructures the text for readability, adds a compelling hook, and includes hashtags or emojis to increase engagement. " \
        "These three agents are orchestrated by a SequentialAgent controller that ensures the workflow runs in the correct order. " \
        "This modular approach not only enforces content quality and safety but also makes the system flexible for future extensions or use cases.", icon="ℹ️")
        st.warning('Currently only tested with Google API', icon="⚠️")
        # --- Disclaimer Box ---
        st.markdown(":orange-badge[⚠️ This content is AI-generated and may not be accurate. Please verify facts before relying on them.]")


    elif st.session_state["page"] == "news_poster":
        st.title("🗞️ AI-powered News Agent")
        query = st.text_input("Enter news topic:", value="AI")
        time_range = st.selectbox("Time range:", ["1h", "6h", "1d"], index=0)

        if st.button("Fetch & Run AI Agent"):
            with st.spinner("Fetching news..."):
                articles = get_the_news(query, time_range)
            if not articles:
                st.error("No articles found for this topic and time range.")
            else:
                csv_columns = ["Headline", "Source", "Posted", "Description", "Link"]
                csv_data = [dict(zip(csv_columns, art)) for art in articles]
                st.subheader("📋 Combined News")
                st.dataframe(csv_data)

                combined = "\n\n".join([f"Headline: {h}\nSource: {s}\nDesc: {d}" for h, s, p, d, l in articles])
                with st.spinner("Running AI agents..."):
                    model = st.session_state["llm_model"]
                    safety, post, state = asyncio.run(run_news_agent(model, combined))

                st.subheader("🔒 Safety Check")
                st.write(safety)
                if safety.strip() == "Safe":
                    st.subheader("📣 Final Post")
                    st.text_area("Post Output", post, height=200)
                else:
                    st.error("Content was flagged as unsafe and will not be posted.", icon="⚠️")

        if st.button("🔄 Reset"):
            st.rerun()

        # --- Disclaimer Box ---
        st.markdown(":orange-badge[⚠️ This content is AI-generated and may not be accurate. Please verify facts before relying on them.]")
if __name__ == "__main__":
    os.environ.clear()
    main()