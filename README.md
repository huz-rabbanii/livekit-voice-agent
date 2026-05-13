# LiveKit AI Voice Agent — Starter Kit

A minimal, well-commented real-time AI voice agent built on [LiveKit Agents](https://docs.livekit.io/agents/).  
Use it as a starting point and layer in your own tools, personas, and integrations.

```
Microphone → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (OpenAI) → Speaker
              ↑
         Silero VAD
```

---

## Features

- **Real-time voice pipeline** — VAD → STT → LLM → TTS in one loop
- **Function/tool calling** — the LLM can call Python functions (e.g. `get_current_time`) mid-conversation
- **Structured logging** — timestamped logs for every session event
- **Swap-friendly** — STT model, LLM model, and TTS voice are constants at the top of `agent.py`
- **Commented placeholders** — weather, web-search, and RAG stubs ready to fill in

---

## Prerequisites

| Requirement | Where to get it |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| LiveKit project | [cloud.livekit.io](https://cloud.livekit.io) → free tier available |
| OpenAI API key | [platform.openai.com](https://platform.openai.com/api-keys) |
| Deepgram API key | [console.deepgram.com](https://console.deepgram.com) → free tier available |

---

## Setup

```bash
# 1. Clone / open the project
cd livekit-ai-voice-agent

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
# Then open .env and fill in your keys
```

### `.env` variables

| Variable | Description |
|---|---|
| `LIVEKIT_URL` | WebSocket URL of your LiveKit server, e.g. `wss://my-project.livekit.cloud` |
| `LIVEKIT_API_KEY` | LiveKit project API key |
| `LIVEKIT_API_SECRET` | LiveKit project API secret |
| `OPENAI_API_KEY` | Used for both the LLM (GPT-4o-mini) and TTS |
| `DEEPGRAM_API_KEY` | Used for speech-to-text (Nova 3) |

---

## Running the agent

### Option A — One-command console (recommended for testing)

```bash
python console.py
```

This will:
1. Generate a participant token from your `.env` credentials
2. Start `agent.py dev` in the background
3. Open the **LiveKit Agents Playground** in your browser with everything pre-filled
4. Shut the agent down cleanly when you press **Ctrl+C**

### Option B — Manual

```bash
# Development mode — connects to a single room for testing
python agent.py dev

# Production worker mode — listens for new rooms and auto-assigns jobs
python agent.py start
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io/),  
paste your LiveKit URL + key/secret, and start talking.

---

## Project structure

```
livekit-ai-voice-agent/
├── agent.py          ← Main agent (edit this)
├── requirements.txt  ← Python dependencies
├── .env.example      ← Credential template
└── .env              ← Your secrets (never commit this)
```

---

## Customising the agent

### Change the persona

Edit `SYSTEM_PROMPT` near the top of `agent.py`:

```python
SYSTEM_PROMPT = """
You are a sarcastic but helpful assistant named Max.
...
"""
```

### Switch models or voices

```python
STT_MODEL = "nova-3"       # or "nova-2", "base", etc.
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o" for higher quality
TTS_VOICE  = "nova"        # alloy | echo | fable | onyx | nova | shimmer
```

### Add a tool

Add a method to `VoiceAgent` and decorate it with `@function_tool`.  
The LLM reads the docstring and type annotations to know when and how to call it.

```python
from livekit.agents import function_tool

class VoiceAgent(Agent):
    ...

    @function_tool
    async def get_weather(self, city: str) -> str:
        """
        Return the current weather for a city.

        Args:
            city: City name, e.g. 'London' or 'Tokyo'.
        """
        # Call your weather API here
        return f"It's 22 °C and sunny in {city}."
```

### Use a different STT provider

```python
# Swap Deepgram for OpenAI Whisper:
from livekit.plugins import openai as lk_openai

session = AgentSession(
    stt=lk_openai.STT(),   # uses Whisper
    ...
)
```

---

## Ideas for what to build next

- **Customer support bot** — load a knowledge base and answer FAQs
- **Language tutor** — converse in a target language and correct mistakes
- **Voice-controlled home automation** — call smart-home APIs as tools
- **Interview coach** — ask practice questions and give spoken feedback
- **Meeting assistant** — join a room, transcribe, and summarise in real time

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LIVEKIT_URL not set` | Ensure `.env` is in the project root and you ran `load_dotenv()` |
| No audio in playground | Check browser mic permissions; try a different browser |
| Agent connects but never speaks | Verify `OPENAI_API_KEY` is valid and has TTS access |
| High latency | Switch to `gpt-4o-mini` (faster) or reduce `SYSTEM_PROMPT` length |

---

## License

MIT — use freely, attribution appreciated.
