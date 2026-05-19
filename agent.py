"""
LiveKit AI Voice Agent — Clinic Appointment Assistant
======================================================
A fully conversational voice agent named "Clara" that handles:
  * Patient registration  (name, email, phone)
  * Appointment booking   (service, provider, date, time)
  * Rescheduling          (change date / time by confirmation ID)
  * Cancellation          (cancel by confirmation ID)
  * Appointment lookup    (list all appointments for a patient)

All data is persisted locally in data.json (no external DB needed).

Pipeline:  Microphone -> VAD (Silero) -> STT (Deepgram) -> LLM (OpenAI) -> TTS (OpenAI) -> Speaker

Quick start:
  1. Copy .env.example -> .env and fill in your API keys.
  2. pip install -r requirements.txt
  3. python agent.py dev          <- connect to a LiveKit room (test via Playground)
     python agent.py start        <- production worker mode
     python agent.py console      <- agent + opens LiveKit Agents Playground (audio-only)
     python agent.py meet         <- agent + opens LiveKit Meet (full UI, great for demo recording)
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
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

# Bootstrap
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice-agent")

# Flat-file data store
DATA_FILE = Path(__file__).parent / "data.json"


def _load_data() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": {}, "appointments": {}}


def _save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Clinic configuration
AVAILABLE_SLOTS = [
    "9:00 AM", "10:00 AM", "11:00 AM",
    "2:00 PM",  "3:00 PM",  "4:00 PM",
]

SERVICES = [
    "General Consultation",
    "Follow-up Visit",
    "Specialist Consultation",
    "Dental Checkup",
    "Eye Exam",
]

PROVIDERS = [
    "Dr. Smith",
    "Dr. Johnson",
    "Dr. Williams",
    "Dr. Brown",
    "Dr. Davis",
]

STT_MODEL = "nova-3"
LLM_MODEL = "gpt-4o-mini"
TTS_VOICE = "nova"

SYSTEM_PROMPT = """
You are Clara, a warm and professional AI receptionist at City Health Clinic.
You assist patients with:
  1. Registering as a new patient (collect name, email, phone)
  2. Booking appointments (service, provider, date, time)
  3. Rescheduling appointments (by confirmation ID)
  4. Cancelling appointments (by confirmation ID)
  5. Viewing their upcoming appointments

Rules you must follow:
- Keep all responses short and conversational - they are spoken aloud.
- Never use markdown, bullet points, numbered lists, or special characters.
- Always confirm details with the patient before finalising any booking, reschedule, or cancellation.
- When listing options, say them naturally: "You can choose Dr. Smith, Dr. Johnson, or Dr. Williams."
- Say dates naturally: "May 20th" not "2026-05-20".
- Say times naturally: "two in the afternoon" or "ten in the morning".
- If a patient wants to book but is not registered, register them first.
- Always ask for the patient's email before taking any account action.
- Use the get_services_and_providers tool whenever a patient asks what is available.
- Use the get_available_slots tool before asking the patient to pick a time.
""".strip()


class VoiceAgent(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def register_patient(self, name: str, email: str, phone: str) -> str:
        """
        Register a new patient in the system.

        Args:
            name:  Full name of the patient.
            email: Email address (used as unique identifier).
            phone: Contact phone number.
        """
        data = _load_data()
        email = email.lower().strip()

        if email in data["users"]:
            existing = data["users"][email]
            return (
                f"A patient named {existing['name']} is already registered with that email. "
                f"No changes were made."
            )

        patient_id = str(uuid.uuid4())[:8].upper()
        data["users"][email] = {
            "id": patient_id,
            "name": name,
            "email": email,
            "phone": phone,
            "registered_at": datetime.now().isoformat(),
        }
        _save_data(data)
        logger.info("Registered patient: %s (%s)", name, email)
        return (
            f"Registration successful! Welcome, {name}. "
            f"Your patient ID is {patient_id}. "
            f"We have your phone number as {phone}."
        )

    @function_tool
    async def lookup_patient(self, email: str) -> str:
        """
        Look up a registered patient by email address.

        Args:
            email: The patient's email address.
        """
        data = _load_data()
        email = email.lower().strip()
        user = data["users"].get(email)
        if not user:
            return (
                f"No patient found with the email {email}. "
                f"They may need to register first."
            )
        return (
            f"Found patient {user['name']}, phone {user['phone']}, "
            f"registered on {user['registered_at'][:10]}."
        )

    @function_tool
    async def get_services_and_providers(self) -> str:
        """
        Return the list of available services and providers at the clinic.
        Call this whenever the patient asks what services or doctors are available.
        """
        services = ", ".join(SERVICES)
        providers = ", ".join(PROVIDERS)
        return (
            f"Available services are: {services}. "
            f"Available providers are: {providers}."
        )

    @function_tool
    async def get_available_slots(self, date: str, provider: str = "") -> str:
        """
        Return available appointment time slots for a given date and optional provider.

        Args:
            date:     The date to check, e.g. 'May 20, 2026'.
            provider: Optional provider name to filter by, e.g. 'Dr. Smith'.
        """
        data = _load_data()
        booked = {
            a["time"]
            for a in data["appointments"].values()
            if a["date"] == date
            and a["status"] == "CONFIRMED"
            and (not provider or a["provider"] == provider)
        }
        available = [s for s in AVAILABLE_SLOTS if s not in booked]
        if not available:
            suffix = f" with {provider}" if provider else ""
            return f"No available slots on {date}{suffix}. Please try a different date."
        slots = ", ".join(available)
        suffix = f" with {provider}" if provider else ""
        return f"Available times on {date}{suffix}: {slots}."

    @function_tool
    async def book_appointment(
        self,
        email: str,
        service: str,
        provider: str,
        date: str,
        time: str,
    ) -> str:
        """
        Book an appointment for a registered patient.
        Always confirm the details with the patient before calling this tool.

        Args:
            email:    Patient's email address (must already be registered).
            service:  Service type, e.g. 'General Consultation'.
            provider: Provider name, e.g. 'Dr. Smith'.
            date:     Appointment date, e.g. 'May 20, 2026'.
            time:     Appointment time, e.g. '10:00 AM'.
        """
        data = _load_data()
        email = email.lower().strip()

        if email not in data["users"]:
            return (
                f"No registered patient found with email {email}. "
                f"Please register the patient first."
            )

        for appt in data["appointments"].values():
            if (
                appt["date"] == date
                and appt["time"] == time
                and appt["provider"] == provider
                and appt["status"] == "CONFIRMED"
            ):
                return (
                    f"{provider} is not available at {time} on {date}. "
                    f"Please choose a different time or provider."
                )

        appt_id = str(uuid.uuid4())[:8].upper()
        patient_name = data["users"][email]["name"]
        data["appointments"][appt_id] = {
            "id": appt_id,
            "user_email": email,
            "user_name": patient_name,
            "service": service,
            "provider": provider,
            "date": date,
            "time": time,
            "status": "CONFIRMED",
            "created_at": datetime.now().isoformat(),
        }
        _save_data(data)
        logger.info("Booked appointment %s for %s", appt_id, email)
        return (
            f"Appointment confirmed! Your confirmation ID is {appt_id}. "
            f"{patient_name} is booked for {service} with {provider} "
            f"on {date} at {time}. Please save your confirmation ID."
        )

    @function_tool
    async def list_appointments(self, email: str) -> str:
        """
        List all upcoming (non-cancelled) appointments for a patient.

        Args:
            email: The patient's email address.
        """
        data = _load_data()
        email = email.lower().strip()
        appts = [
            a
            for a in data["appointments"].values()
            if a["user_email"] == email and a["status"] != "CANCELLED"
        ]
        if not appts:
            return f"There are no upcoming appointments for {email}."
        parts = [
            f"ID {a['id']}: {a['service']} with {a['provider']} on {a['date']} at {a['time']}, status {a['status']}"
            for a in appts
        ]
        return "Here are the upcoming appointments: " + ". Next: ".join(parts) + "."

    @function_tool
    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_date: str,
        new_time: str,
    ) -> str:
        """
        Reschedule an existing confirmed appointment to a new date and time.
        Always confirm the new date and time with the patient before calling this tool.

        Args:
            appointment_id: The 8-character confirmation ID, e.g. 'A3F9B2C1'.
            new_date:       New appointment date, e.g. 'May 25, 2026'.
            new_time:       New appointment time, e.g. '3:00 PM'.
        """
        data = _load_data()
        appt_id = appointment_id.upper().strip()

        if appt_id not in data["appointments"]:
            return (
                f"No appointment found with ID {appt_id}. "
                f"Please double-check the confirmation ID."
            )

        appt = data["appointments"][appt_id]
        if appt["status"] == "CANCELLED":
            return f"Appointment {appt_id} has already been cancelled and cannot be rescheduled."

        for other in data["appointments"].values():
            if (
                other["id"] != appt_id
                and other["date"] == new_date
                and other["time"] == new_time
                and other["provider"] == appt["provider"]
                and other["status"] == "CONFIRMED"
            ):
                return (
                    f"{appt['provider']} is not available at {new_time} on {new_date}. "
                    f"Please choose a different time."
                )

        old_date, old_time = appt["date"], appt["time"]
        appt["date"] = new_date
        appt["time"] = new_time
        appt["rescheduled_at"] = datetime.now().isoformat()
        _save_data(data)
        logger.info("Rescheduled appointment %s to %s %s", appt_id, new_date, new_time)
        return (
            f"Done! Appointment {appt_id} has been rescheduled "
            f"from {old_date} at {old_time} "
            f"to {new_date} at {new_time}."
        )

    @function_tool
    async def cancel_appointment(self, appointment_id: str) -> str:
        """
        Cancel an existing appointment.
        Always ask the patient to confirm before calling this tool.

        Args:
            appointment_id: The 8-character confirmation ID, e.g. 'A3F9B2C1'.
        """
        data = _load_data()
        appt_id = appointment_id.upper().strip()

        if appt_id not in data["appointments"]:
            return f"No appointment found with ID {appt_id}. Please check the confirmation ID."

        appt = data["appointments"][appt_id]
        if appt["status"] == "CANCELLED":
            return f"Appointment {appt_id} is already cancelled."

        appt["status"] = "CANCELLED"
        appt["cancelled_at"] = datetime.now().isoformat()
        _save_data(data)
        logger.info("Cancelled appointment %s", appt_id)
        return (
            f"Appointment {appt_id} for {appt['service']} with {appt['provider']} "
            f"on {appt['date']} at {appt['time']} has been successfully cancelled."
        )

    @function_tool
    async def get_current_date_and_time(self, timezone: str = "UTC") -> str:
        """
        Return the current date and time. Useful when the patient wants to book
        for 'today' or 'tomorrow', or asks what time it is.

        Args:
            timezone: IANA timezone name, e.g. 'America/New_York'. Defaults to UTC.
        """
        try:
            tz = ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")
        now = datetime.now(tz)
        hour = int(now.strftime("%I"))
        rest = now.strftime(":%M %p on %A, %B %d, %Y")
        return f"{hour}{rest}"


async def entrypoint(ctx: JobContext) -> None:
    """Called by the LiveKit worker for every new room / participant."""
    logger.info("Agent connecting to room: %s", ctx.room.name)
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=STT_MODEL),
        llm=openai.LLM(model=LLM_MODEL),
        tts=openai.TTS(voice=TTS_VOICE),
    )

    await session.start(agent=VoiceAgent(), room=ctx.room)
    logger.info("Session started — sending greeting.")

    await session.generate_reply(
        instructions=(
            "Greet the patient warmly. Introduce yourself as Clara, the voice receptionist "
            "at City Health Clinic. Tell them you can help with registration, booking "
            "appointments, rescheduling, and cancellations. Ask how you can help them today."
        )
    )


def _launch_browser_mode(open_meet: bool = False) -> None:
    """
    Shared launcher for console and meet modes.

    console  ->  opens https://agents-playground.livekit.io  (audio-only, agent-focused)
    meet     ->  opens https://meet.livekit.io               (full video UI, ideal for demos)
    """
    ROOM_NAME            = "clinic-demo"
    PARTICIPANT_IDENTITY = "demo-patient"
    PARTICIPANT_NAME     = "Demo Patient"
    TOKEN_TTL            = 3600
    STARTUP_DELAY        = 2.5

    livekit_url = os.environ.get("LIVEKIT_URL", "").strip()
    api_key     = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret  = os.environ.get("LIVEKIT_API_SECRET", "").strip()

    for var, val in [
        ("LIVEKIT_URL",        livekit_url),
        ("LIVEKIT_API_KEY",    api_key),
        ("LIVEKIT_API_SECRET", api_secret),
    ]:
        if not val:
            print(f"[launcher] ERROR: '{var}' is not set. Check your .env file.")
            sys.exit(1)

    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        print("[launcher] ERROR: 'livekit-api' is not installed.\n  Run: pip install livekit-api")
        sys.exit(1)

    print("[launcher] Generating participant token ...")
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(PARTICIPANT_IDENTITY)
        .with_name(PARTICIPANT_NAME)
        .with_grants(VideoGrants(room_join=True, room=ROOM_NAME))
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )

    if open_meet:
        browser_url = (
            f"https://meet.livekit.io"
            f"?roomName={quote(ROOM_NAME, safe='')}"
            f"&liveKitUrl={quote(livekit_url, safe='')}"
            f"&token={token}"
        )
        mode_label = "LiveKit Meet  (great for screen-recording a demo)"
    else:
        browser_url = (
            f"https://agents-playground.livekit.io"
            f"/#url={quote(livekit_url, safe='')}&token={token}"
        )
        mode_label = "LiveKit Agents Playground"

    print(f"[launcher] Starting agent worker (room: {ROOM_NAME}) ...")
    agent_proc = subprocess.Popen([sys.executable, __file__, "dev", "--room", ROOM_NAME])

    print(f"[launcher] Waiting {STARTUP_DELAY}s for agent to connect ...")
    time.sleep(STARTUP_DELAY)

    print("\n" + "=" * 64)
    print(f"  {mode_label}")
    print("=" * 64)
    print(f"  Room     : {ROOM_NAME}")
    print(f"  Identity : {PARTICIPANT_IDENTITY}")
    print("=" * 64)
    print("  Browser opening ... press Ctrl+C to stop the agent.\n")
    webbrowser.open(browser_url)

    try:
        agent_proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down agent ...")
        if sys.platform == "win32":
            agent_proc.send_signal(signal.CTRL_C_EVENT)
        else:
            agent_proc.terminate()
        try:
            agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
        print("[launcher] Agent stopped. Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        sys.argv.pop(1)
        _launch_browser_mode(open_meet=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "meet":
        sys.argv.pop(1)
        _launch_browser_mode(open_meet=True)
    else:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
