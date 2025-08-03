import os
from twilio.rest import Client
from dotenv import load_dotenv

# --- Configuration and Setup ---

# Load environment variables from a .env file
# Example .env file content:
# GEMINI_API_KEY="your_gemini_api_key"
# TWILIO_ACCOUNT_SID="your_twilio_account_sid"
# TWILIO_AUTH_TOKEN="your_twilio_auth_token"
# TWILIO_PHONE_NUMBER="+1234567890"
# MY_NUMBER="+19876543210"

load_dotenv()

# 1. Load credentials from environment variables
try:
    twilio_account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    twilio_auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    twilio_phone_number = os.environ["TWILIO_PHONE_NUMBER"]
    my_phone_number = os.environ["MY_NUMBER"]
except KeyError as e:
    print(f"Error: Environment variable {e} not found.")
    print("Please make sure your .env file is set up correctly.")
    exit()

# 2. Initialize the Twilio Client
try:
    twilio_client = Client(twilio_account_sid, twilio_auth_token)
except Exception as e:
    print(f"Error initializing Twilio client: {e}")
    exit()

# --- Hospital Visit Message ---

KNOWLEDGE_BASE = """Hello! This is your healthcare appointment confirmation call.

Patient Name: Patient
Department: ** Orthopedics
Doctor: ** Dr. Robert Johnson
Date: ** August 13, 2025
Time: ** 03:00 PM
Location: ** Building B, Floor 1, Room 101-110

Please arrive 15 minutes before your appointment time. Bring your ID and insurance card. 
For questions, call the hospital at 555-123-4567.

Thank you for choosing our healthcare system!"""

# --- Generate and Make the Call ---

def generate_dynamic_message():
    print("Returning static hospital guidance message...")
    return KNOWLEDGE_BASE

def make_phone_call(message_to_say):
    if not message_to_say:
        print("Cannot make a call with an empty message.")
        return

    print(f"Initiating call to {my_phone_number}...")
    try:
        # Clean up the message to remove line breaks that may affect speech
        cleaned_message = ' '.join(message_to_say.strip().splitlines())

        twiml_instruction = f'<Response><Say voice="alice">{cleaned_message}</Say></Response>'

        call = twilio_client.calls.create(
            twiml=twiml_instruction,
            to=my_phone_number,
            from_=twilio_phone_number
        )
        print(f"Call initiated successfully! Call SID: {call.sid}")
        print("Your phone should be ringing shortly.")

    except Exception as e:
        print(f"Error making phone call with Twilio: {e}")

# --- Execution ---

if __name__ == "__main__":
    dynamic_message = generate_dynamic_message()
    make_phone_call(dynamic_message)
