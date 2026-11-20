# Dream Detective — Streamlit project

A sleep-and-consciousness mystery game. Players ask natural-language yes/no questions to identify a hidden sleep phenomenon. The LLM maps player wording to a controlled feature vocabulary; Python retrieves the answer from a transparent case knowledge base and computes question information gain.

## Features
- Two pages: **Play** and **Stats**
- Streamlit chat interface (`st.chat_message`, `st.chat_input`)
- LLM-assisted natural-language interpretation
- Deterministic answer checking from a case fact table
- Guess/question counts, per-game bar chart, and transparent question-quality estimate
- Graceful fallback when the LLM endpoint is not configured or unavailable

## 1. Requirements
- Python 3.10+
- An API key from the university's AI key service
- The OpenAI-compatible API base URL and model name shown by that service

## 2. Run locally (Windows)
Open this folder in VS Code. In the terminal:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` by copying `secrets.example.toml`, then fill in the values provided by the university AI key portal:

```toml
AI_API_KEY = "paste-your-key-here"
AI_BASE_URL = "paste-the-OpenAI-compatible-base-URL-here"
AI_MODEL = "paste-the-model-name-here"
```

Do not upload `secrets.toml` or share your API key.

Run:

```bash
streamlit run app.py
```

If the API is not configured, the game still runs using a small keyword-based fallback, but the LLM requirement will not be active.

## 3. Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Open https://share.streamlit.io and connect your GitHub account.
3. Create an app, choose this repository, branch `main`, and file `app.py`.
4. In **Advanced settings → Secrets**, paste the three TOML settings above.
5. Deploy and copy the public `streamlit.app` URL for the assignment.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 4. Project files
- `app.py` — complete Streamlit app
- `requirements.txt` — Python dependencies
- `secrets.example.toml` — safe configuration template (no real key)
- `.gitignore` — prevents secrets and local files from being committed
- `tests/test_game_logic.py` — small tests for information-gain scoring

## 5. Important design note
The question-quality score is an educational game metric. It measures the expected information gain of the feature being asked about across the six cases in this archive. It is not a validated measure of intelligence or general reasoning ability.

## 6. Before submission
- Configure the university API endpoint and model.
- Test several cases and malformed inputs.
- Verify the public deployment opens in an incognito/private browser.
- Add your actual contribution statement; do not claim work that you did not do.
