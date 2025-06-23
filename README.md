# 📰 AI-Powered News Agent 🚀

**Built by [Sanjeev Mohan](mailto:mysanjeev99@gmail.com), Balaharidass Ramachandran, and Karthikeyan S.**

This project is an intelligent **multi-agent system** that fetches recent news articles from Yahoo News based on a topic and time range, summarizes them using an LLM, filters for safety, and formats the result into a platform-ready social media post.

---

## 📌 Features

* 🔍 Scrape real-time news articles from Yahoo News
* 🤖 Summarize articles into professional news drafts using LLMs
* 🛡️ Strict content safety filtering
* 💬 Auto-format into engaging social media posts (with hashtags/emojis)
* 🧠 Multi-agent pipeline with Google ADK
* 💍 Streamlit UI for interactive exploration
* 🔑 Supports multiple LLMs: Google Gemini, OpenAI GPT, Anthropic Claude

---

## 🧰 Tech Stack

* **Frontend/UI**: Streamlit
* **Web Scraping**: BeautifulSoup, requests
* **LLM Integration**: Google ADK (Agents Development Kit), Gemini, GPT, Claude
* **Backend Logic**: Python asyncio, multi-agent orchestration
* **Deployment**: Local or cloud (e.g., AWS, Streamlit Cloud)

---

## ⚙️ Installation

### Prerequisites

* Python 3.9+
* API Keys for:

  * [Google Generative AI](https://makersuite.google.com/app)
  * [Google Vertex AI](https://cloud.google.com/vertex-ai)
  * [OpenAI](https://platform.openai.com)
  * [Anthropic](https://console.anthropic.com/)

### Clone and Install Dependencies

```bash
git clone https://github.com/mysanjeev99/social-media-agent.git
cd social-media-agent
pip install -r requirements.txt
```

> Note: Make sure you have access to Google ADK packages (`google.adk`). This may require private/internal access.

---

## 🚀 Run the App

```bash
streamlit run app_streamlit.py
```

On launch, you’ll be prompted to:

1. Select an LLM provider and model.
2. Paste the corresponding API key.
3. Click **"Start App"** to proceed to the news agent interface.

---

## 🧠 Architecture: Multi-Agent Pipeline

This app leverages a **modular multi-agent architecture** using the Google Agents Development Kit (ADK). Each agent in the sequence performs a distinct responsibility:

### 1. `NewsGenerator`

* Summarizes multiple raw news articles into a single coherent, fact-based summary (\~250 words)
* Journalistic tone, no opinions or filler

### 2. `ContentFilter`

* Evaluates the summary for public safety and compliance
* Detects political bias, hate speech, misinformation, or unsafe language
* Responds strictly with `"Safe"` or `"Unsafe"`

### 3. `PostFormatter`

* If content is "Safe", converts the summary into an engaging social media post
* Adds a compelling hook, formatting, emojis, and hashtags
* Suitable for platforms like LinkedIn, Twitter

The entire pipeline is managed via a `SequentialAgent` that orchestrates flow from input to output.

---

## 🧪 Example Use Case

1. **Topic**: `"Artificial Intelligence"`
2. **Time Range**: `"1h"`
3. **Result**:

   * The app scrapes the latest news about AI from Yahoo
   * LLM generates a 250-word summary
   * Safety check passes ✅
   * Final post is formatted:

     ```
     🤖 The AI world just got more exciting...
     [Summary Content Here]
     #AI #TechNews #LLM
     ```

---

## ⚠️ Disclaimer

> **This content is AI-generated and may not be accurate.**
> Please verify all facts before relying on or sharing generated outputs.

---

## 👨‍💼 Contributors

* **Sanjeev Mohan** – [LinkedIn](https://www.linkedin.com/in/sanjeev-mohan/)
* **Balaharidass Ramachandran**
* **Karthikeyan S**

---

## 🏃‍♂️ Contact

Feel free to reach out with questions or collaboration ideas:
📧 [mysanjeev99@gmail.com](mailto:mysanjeev99@gmail.com)