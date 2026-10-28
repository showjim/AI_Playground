# AGENTS.md

Jerry's AI Playground is a Streamlit multipage app for AI experiments: chat,
RAG over documents, meeting minutes, and RAG evaluation. `home_page.py` is the
entry point; each file in `pages/` is a standalone mini-app.

## Commands

Run from the repo root with Python 3.11 (`.venv` already exists):

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run home_page.py                         # full app
python -m streamlit run "pages/2_💬_CasualChat.py" # one page
PYTHONPATH=. .venv/bin/python test/<script>.py     # scratch script
```

No pytest, linter, or formatter is configured; `test/` holds manual experiments,
not tests. The `ffmpeg` binary must be installed separately.

## Rules

**Think first.** State your assumptions. Ask when the request has more than one
reasonable reading. If a simpler approach exists, say so before coding.

**Simplicity First — no over-coding.** Write the smallest change that solves the
task. Use one concrete implementation until a second caller needs the
abstraction. Add options, configurability, and fallbacks when the task asks for
them. Add a dependency only when the standard library cannot do the job, and ask
first. Refactoring working code is a separate task, not a side effect.

**Surgical Changes.** Touch only the lines the task needs. Match the surrounding
style. Clean up only what your own edit left unused; report unrelated dead code
instead of deleting it. Every changed line should trace back to the request.

**Verify.** There is no automated test suite. Run the affected page or script
and say what you ran. If you cannot run it because a key, service, or model is
missing, say that plainly.

## Project facts

- Streamlit only discovers `pages/`; `deprecated/` is hidden. The
  `<number>_<emoji>_<Name>.py` prefix sets sidebar order.
- Two backend layers, and each page already picks one:
  - LangChain: `src/llms.py` and `src/chat.py` (Azure OpenAI, Ollama, FAISS).
  - Native SDK: `src/ClsChatBot.py` (Azure, OpenRouter, SiliconFlow, Speech).
  Keep a change inside the layer the page already uses.
- Pages resolve paths from the repo root (`os.path.abspath('.')`), so run every
  command from the root.
- `key.txt` holds dotenv-format secrets and is copied to `.env` by every
  `setup_env()`; edit `key.txt`, not `.env`. `config.json` holds non-secret
  settings. `.streamlit/secrets.toml` holds the OpenChat password. `key.txt` and
  `.streamlit/secrets.toml` contain live credentials and are tracked in git;
  don't print their contents.
- Runtime paths: `workDir/` uploads, `index/` and `Output/` FAISS indexes,
  `tempDir/` transcripts and eval uploads, `setting/*.db` chat history, `img/`
  avatars and uploads. Create `tempDir/` before the meeting-minutes page writes
  to it; FAISS creates its own index folders.
- Model dispatch in `src/llms.py` is substring-based (`"gpt"` or `"ollama"`),
  and each page sidebar hard-codes its model list. Add a provider in
  `src/llms.py`, not in a page.
- `requirements.txt` does not cover every page: `evaluate` (page 7),
  `llama-index-*` (page 14), and `htmlmin` (`src/html_preprocess.py`). Point
  this out when a task needs them.
- Known stale: page 1 imports the removed `ChatRobot` class; page 13 reads
  `./setting/key.txt` and `./setting/config.json`, which do not exist. Page 10
  (FreeChat) and `ChatRobotAzure` are the working equivalents.
- The Dockerfile exposes and health-checks port 8504; Streamlit defaults to
  8501 unless `--server.port 8504` is passed.

## Communication

Use plain language: short sentences, everyday words. Define a term the first
time you use it, and explain why before how. Lead with the answer, then the
details. If something is complex or uncertain, say so directly.
