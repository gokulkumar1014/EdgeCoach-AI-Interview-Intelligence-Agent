# EdgeCoach AI - A Retrieval Augmented Interview Intelligence Assistant

EdgeCoach AI is a lightweight, serverless interview-intelligence assistant built for students who receive interview invitations with short notice and need **company- and role-specific** preparation quickly.

The system:
- Detects when a user wants *interview intel* vs. general Q&A.
- Extracts company, role, and time-to-interview from free-text queries.
- Retrieves real candidate experiences from the public web.
- Synthesizes a structured prep guide (rounds, themes, plan, takeaways).
- Surfaces original web sources for transparency.

This project was developed for the **Text Analytics** course (MSBA, The George Washington University, Fall 2025).

---

## 1. Tech Stack

- **Language:** Python 3.10/3.11  
- **Frontend:** Streamlit (chat UI)  
- **Backend:** AWS Lambda (Function URL)  
- **LLM Platform:** Amazon Bedrock – Anthropic Claude models  
- **Retrieval:** Tavily Search API  
- **Web Extraction:** `requests`, `trafilatura`, `beautifulsoup4`, `PyPDF2`  
- **AWS SDK:** `boto3`

---

## 2. Repository Structure

```text
INTERVIEW-EXP-AGENT/
├─ app.py                     # Streamlit frontend (EdgeCoach UI)
├─ lambda.zip                 # Deployment bundle for AWS Lambda (optional artifact)
├─ venv/                      # Local virtual environment (ignored in deployment)
└─ lambda/
   ├─ handler.py              # AWS Lambda entrypoint
   ├─ bedrock_intent.py       # Intent classification via Bedrock (Claude)
   ├─ tavily_retrieval.py     # Tavily search + web scraping & cleaning
   ├─ analysis_engine.py      # RAG synthesis & EdgeCoach persona
   ├─ requests/, certifi/, ...   # Vendored third-party dependencies
```

You may also have additional packaging/metadata files such as `requirements.txt` or `README.md`.

---

## 3. High-Level Architecture

**Frontend (Streamlit)**  
- `app.py` exposes a chat interface.  
- Sends `{ query, messages }` to the Lambda HTTP endpoint (`API_URL` in `app.py`).  
- Renders assistant replies and, when interview intel is requested, expandable cards showing the web sources used.

**Backend (AWS Lambda – `handler.py`)**

1. Parses the HTTP event body (`query`, `messages`).  
2. Restores cached agent state (company, role, sources) from a hidden system message.  
3. Calls the **Intent Module** (`bedrock_intent.py`) to decide:
   - Is this an interview-prep request?
   - What company, role, and time-to-interview did the user mention?
4. If it’s general Q&A → routes to a small Bedrock call for a standard assistant answer.  
5. If the user wants interview intel:
   - Calls the **Tavily Search Module** (`tavily_retrieval.py`) to pull candidate experiences from the web and clean them.
   - Calls the **Analysis Engine** (`analysis_engine.py`) to build a structured five-section prep guide.
6. Returns `intent`, `answer`, `sources`, and updated `messages` back to the frontend.

**Data & External Services**

- **Tavily Search API** → web search over public interview content.  
- **Public web sources** → Glassdoor, Reddit, Blogs, Linkedin PDFs, etc.  
- **Amazon Bedrock (Claude)** → used for intent extraction and answer synthesis.

---

## 4. Setup & Local Development

### 4.1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # on macOS / Linux
venv\Scripts\activate    # on Windows
```

### 4.2. Install dependencies

Create a `requirements.txt` similar to:

```txt
streamlit
boto3
requests
trafilatura
beautifulsoup4
PyPDF2
```

Then install:

```bash
pip install -r requirements.txt
```

### 4.3. Environment variables

Set the following environment variables before running locally:

- `TAVILY_API_KEY` - API key for Tavily Search.  
- `BEDROCK_INTENT_MODEL_ID` - optional override for the intent model (default: `anthropic.claude-3-haiku-20240307-v1:0`).
- `BEDROCK_ANALYSIS_MODEL_ID` - optional override for the analysis model (default: `anthropic.claude-3-haiku-20240307-v1:0`).
- AWS credentials configured via `aws configure` or environment variables so that `boto3` can call Bedrock.

---

## 5. Running the Streamlit Frontend Locally

From the project root:

```bash
streamlit run app.py
```

The app will open in your browser (typically at `http://localhost:8501`).  
You can chat with EdgeCoach AI, and the frontend will call the configured Lambda URL defined in `API_URL` inside `app.py`.

> **Note:** For fully local testing without deploying to AWS, you can temporarily replace `API_URL` with `http://localhost:8000/...` and point it to a local FastAPI/Lambda emulator, but in our course deployment we use the real AWS Lambda Function URL.

---

## 6. Deploying the Lambda Backend (High-Level)

1. Create a Python 3.x Lambda function in AWS.  
2. Package the contents of the `lambda/` folder (including `requests`, `certifi`, etc.) into `lambda.zip` with `handler.lambda_handler` as the entrypoint.  
3. Configure environment variables (`TAVILY_API_KEY`, `BEDROCK_INTENT_MODEL_ID` if needed).  
4. Attach an IAM role that allows `bedrock:InvokeModel` and basic CloudWatch logging.  
5. Create a **Function URL** for the Lambda and paste it into `API_URL` in `app.py`.

---

## 7. Code Organization

- **`handler.py`** – orchestrates the whole flow:
  - Parses requests, manages state, routes between general Q&A and interview mode.
- **`bedrock_intent.py`** – calls Claude on Bedrock to get JSON intent; falls back to regex heuristics if needed.
- **`tavily_retrieval.py`** – issues Tavily queries, deduplicates URLs, fetches pages, and extracts clean text.
- **`analysis_engine.py`** – builds a structured prompt with candidate profile + sources and asks Claude to generate a five-part prep guide.
- **`app.py`** – Streamlit UI, manages chat history and renders answers and source cards.

---

## 8. Use of Generative AI Tools

During this project we used generative AI tools (including ChatGPT and Codex-style assistants) to help with brainstorming, code drafting, and editing of written text. All model outputs were reviewed, tested, and revised by the project team; we remain fully responsible for the correctness of the code, analyses, and conclusions presented.

---

## 9. Acknowledgements

This project was completed as part of the **Text Analytics** course in the MSBA program at The George Washington University. We thank the instructor and classmates for feedback and suggestions throughout the semester.
