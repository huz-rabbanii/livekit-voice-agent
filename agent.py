"""
LiveKit AI Voice Agent — Starter Kit
=====================================
A ready-to-run conversational voice agent built on LiveKit Agents.

Pipeline:  Microphone → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (OpenAI) → Speaker
                         ↑
                    (Silero VAD)

Quick start:
  1. Copy .env.example → .env and fill in your API keys.
  2. pip install -r requirements.txt
  3. python agent.py dev          ← connects to a LiveKit room for testing
     python agent.py start        ← production worker mode
     python agent.py console      ← starts agent + opens browser playground

Adding tools:  Add a method decorated with @function_tool inside VoiceAgent.
               The LLM will automatically discover and call it when relevant.
"""

import logging
import os
import signal
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, openai, silero

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice-agent")

# ── System prompt ──────────────────────────────────────────────────────────────
# Edit this to change the agent's personality, language, or focus area.
SYSTEM_PROMPT = """
You are a helpful, friendly AI voice assistant.
Keep your responses concise and conversational — they will be spoken aloud.
Avoid markdown, bullet points, numbered lists, or any special formatting.
When you use a tool, weave the result naturally into your spoken reply.
If you don't know something, say so honestly rather than guessing.
""".strip()

# ── Model & voice configuration ────────────────────────────────────────────────
# Swap these out to use different providers or models.
STT_MODEL  = "nova-3"          # Deepgram STT model
LLM_MODEL  = "gpt-4o-mini"    # OpenAI chat model  (gpt-4o for higher quality)
TTS_VOICE  = "alloy"           # OpenAI TTS voice   (alloy | echo | fable | onyx | nova | shimmer)


# ── Agent ──────────────────────────────────────────────────────────────────────
class VoiceAgent(Agent):
    """
    Starter voice agent.

    To add a new capability, define a method here and decorate it with
    @function_tool.  The docstring becomes the tool description the LLM sees,
    so make it clear and descriptive.  Type-annotate every parameter so the
    LLM knows what to pass.
    """

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    # ── Built-in tools ─────────────────────────────────────────────────────────

    @function_tool
    async def get_current_time(self, timezone: str = "UTC") -> str:
        """
        Return the current date and time.

        Args:
            timezone: A valid IANA timezone name, e.g. 'America/New_York' or
                      'Europe/London'.  Defaults to UTC when omitted.
        """
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            logger.warning("Unknown timezone '%s', falling back to UTC.", timezone)
            tz = ZoneInfo("UTC")

        now = datetime.now(tz)
        # Cross-platform readable format: "2:05 PM on Tuesday, May 13, 2025"
        hour = int(now.strftime("%I"))          # strip leading zero
        rest = now.strftime(":%M %p on %A, %B %d, %Y")
        return f"{hour}{rest}"

    # ── Add your own tools below ───────────────────────────────────────────────
    #
    # Example — uncomment and fill in the TODO to enable a weather tool:
    #
    # @function_tool
    # async def get_weather(self, city: str) -> str:
    #     """
    #     Return the current weather for a given city.
    #
    #     Args:
    #         city: The city name to look up, e.g. 'London' or 'Tokyo'.
    #     """
    #     # TODO: call a real weather API (e.g. Open-Meteo, OpenWeatherMap)
    #     return f"I don't have live weather data yet, but {city} sounds lovely!"
    #
    # ── More ideas ─────────────────────────────────────────────────────────────
    # • search_web(query)         — fetch a quick search summary
    # • set_reminder(message, minutes_from_now)
    # • answer_from_docs(question) — RAG over your own documents
    # ──────────────────────────────────────────────────────────────────────────


# ── Session entrypoint ─────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext) -> None:
    """Called by the LiveKit worker for every new room/participant."""
    logger.info("Agent connecting to room: %s", ctx.room.name)
    await ctx.connect()

    session = AgentSession(
        # Voice Activity Detection — detects when the user starts/stops speaking
        vad=silero.VAD.load(),

        # Speech-to-Text — transcribes the user's audio
        stt=deepgram.STT(model=STT_MODEL),

        # Large Language Model — generates the reply
        llm=openai.LLM(model=LLM_MODEL),

        # Text-to-Speech — speaks the reply aloud
        tts=openai.TTS(voice=TTS_VOICE),
    )

    await session.start(agent=VoiceAgent(), room=ctx.room)
    logger.info("Session started — sending greeting.")

    # Opening message spoken to the user when they join
    await session.generate_reply(
        instructions=(
            "Greet the user warmly, introduce yourself as an AI voice assistant, "
            "and ask how you can help them today."
        )
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def _run_console() -> None:
    """
    `python agent.py console`
    Generates a participant token, starts the agent in dev mode,
    and opens the LiveKit Agents Playground in the browser.
    """
    ROOM_NAME            = "console-room"
    PARTICIPANT_IDENTITY = "console-user"
    PARTICIPANT_NAME     = "Console User"
    TOKEN_TTL            = 3600          # seconds
    STARTUP_DELAY        = 2.0           # wait before opening browser
    PLAYGROUND_BASE      = "https://agents-playground.livekit.io/"

    livekit_url = os.environ.get("LIVEKIT_URL", "").strip()
    api_key     = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret  = os.environ.get("LIVEKIT_API_SECRET", "").strip()

    for name, val in [("LIVEKIT_URL", livekit_url), ("LIVEKIT_API_KEY", api_key), ("LIVEKIT_API_SECRET", api_secret)]:
        if not val:
            print(f"[console] ERROR: '{name}' is not set. Check your .env file.")
            sys.exit(1)

    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        print("[console] ERROR: 'livekit-api' is not installed.\n  Run: pip install livekit-api")
        sys.exit(1)

    print("[console] Generating participant token …")
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(PARTICIPANT_IDENTITY)
        .with_name(PARTICIPANT_NAME)
        .with_grants(VideoGrants(room_join=True, room=ROOM_NAME))
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )

    playground_url = f"{PLAYGROUND_BASE}#url={quote(livekit_url, safe='')}&token={token}"

    print(f"[console] Starting agent (room: {ROOM_NAME}) …")
    agent_proc = subprocess.Popen([sys.executable, __file__, "dev", "--room", ROOM_NAME])

    print(f"[console] Waiting {STARTUP_DELAY}s for agent to connect …")
    time.sleep(STARTUP_DELAY)

    print("\n" + "=" * 60)
    print("  LiveKit Agents Playground")
    print("=" * 60)
    print(f"  Room     : {ROOM_NAME}")
    print(f"  Identity : {PARTICIPANT_IDENTITY}")
    print("=" * 60)
    print("  Opening browser … (Ctrl+C to stop)\n")
    webbrowser.open(playground_url)

    try:
        agent_proc.wait()
    except KeyboardInterrupt:
        print("\n[console] Shutting down agent …")
        if sys.platform == "win32":
            agent_proc.send_signal(signal.CTRL_C_EVENT)
        else:
            agent_proc.terminate()
        try:
            agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
        print("[console] Agent stopped. Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        sys.argv.pop(1)          # remove 'console' so LiveKit CLI doesn't see it
        _run_console()
    else:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
