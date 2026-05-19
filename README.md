# LiveKit AI Voice Agent — Clinic Appointment Assistant

A fully conversational real-time voice agent named **Clara** built on [LiveKit Agents](https://docs.livekit.io/agents/).  
Clara acts as a clinic receptionist and handles end-to-end patient interactions entirely by voice.

```
Microphone -> VAD (Silero) -> STT (Deepgram) -> LLM (OpenAI gpt-4o-mini) -> TTS (OpenAI nova) -> Speaker
```

---

## What Clara can do

| Capability | What to say |
|---|---|
| **Register** a new patient | "I'd like to register" / "Sign me up" |
| **Book** an appointment | "I need an appointment" / "Book me with Dr. Smith" |
| **View** appointments | "What appointments do I have?" |
| **Reschedule** an appointment | "Move my appointment to Friday at 3 PM" |
| **Cancel** an appointment | "Cancel my appointment ID A3F9B2C1" |

All data is stored in `data.json` (auto-created, no database needed).

---

## Prerequisites

| Requirement | Where to get it |
|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| LiveKit project | [cloud.livekit.io](https://cloud.livekit.io) — free tier available |
| OpenAI API key | [platform.openai.com](https://platform.openai.com/api-keys) |
| Deepgram API key | [console.deepgram.com](https://console.deepgram.com) — free tier available |

---

## Setup

```bash
# 1. Clone / open the project
cd livekit-ai-voice-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY, DEEPGRAM_API_KEY
```

---

## Running the agent

### Option 1 — `console` (Agents Playground, audio-only)
Best for quick testing without video.

```bash
python agent.py console
```
Automatically generates a token, starts the worker, and opens **https://agents-playground.livekit.io** in your browser.  
Click the microphone button and start talking to Clara.

---

### Option 2 — `meet` (LiveKit Meet — full video UI, best for recording a demo)

```bash
python agent.py meet
```
Same as console but opens **https://meet.livekit.io** instead.  
You get a full video-conference UI — ideal for screen-recording a demo.

---

### Option 3 — `dev` (headless worker, bring your own UI)

```bash
python agent.py dev
```
Starts the worker and waits for participants to join any room.  
Connect from the [Agents Playground](https://agents-playground.livekit.io) manually with your own token.

---

### Option 4 — `start` (production)

```bash
python agent.py start
```

---

## Example conversation

```
Clara:  Hi! I'm Clara, the voice receptionist at City Health Clinic.
        I can help you register, book appointments, reschedule, or cancel.
        How can I help you today?

You:    I'd like to book an appointment.

Clara:  Of course! Could I get your email address to look up your account?

You:    I'm not registered yet.

Clara:  No problem! I'll register you first. What's your full name?

You:    John Smith.

Clara:  And your email address?

You:    john@example.com

Clara:  Great. And your phone number?

You:    555-1234

Clara:  Perfect, you're registered, John! Now for your appointment —
        we offer General Consultation, Follow-up Visit, Specialist Consultation,
        Dental Checkup, and Eye Exam. Which would you like?

You:    General Consultation with Dr. Smith on May 25th.

Clara:  Let me check availability... Dr. Smith has openings at
        9 AM, 10 AM, 11 AM, 2 PM, 3 PM, and 4 PM on May 25th.
        Which time works for you?

You:    10 in the morning.

Clara:  Just to confirm: General Consultation with Dr. Smith on May 25th at 10 AM.
        Shall I go ahead and book that?

You:    Yes please.

Clara:  Appointment confirmed! Your confirmation ID is A3F9B2C1.
        Please save that ID in case you need to reschedule or cancel.
```

---

## Project structure

```
livekit-ai-voice-agent/
  agent.py          <- Main agent (all tools + launcher logic)
  data.json         <- Auto-created: stores patients and appointments
  requirements.txt
  .env.example
  .env              <- Your secrets (gitignored)
```

---

## Customising

- **Change the clinic name / persona** — edit `SYSTEM_PROMPT` in `agent.py`
- **Add/remove services or providers** — edit `SERVICES` and `PROVIDERS` lists
- **Add time slots** — edit `AVAILABLE_SLOTS`
- **Swap LLM / STT / TTS** — change `LLM_MODEL`, `STT_MODEL`, `TTS_VOICE` constants
- **Add a new tool** — define an `async def` method on `VoiceAgent` with `@function_tool`
