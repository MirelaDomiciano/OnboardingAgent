from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime
import os.path
import pickle

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def get_credentials():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def create_event(summary, location, description, start_time, end_time, timezone='America/Sao_Paulo'):
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': timezone,
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': timezone,
        },
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {event.get('htmlLink')}"

def create_event_tool(inputs):
    if isinstance(inputs, str):
        import json
        try:
            inputs = json.loads(inputs)
        except json.JSONDecodeError as e:
            return f"Error decoding input string to dictionary: {str(e)}"
    
    if not isinstance(inputs, dict):
        return "Error: Input should be a dictionary."
    
    try:
        summary = inputs.get('summary')
        location = inputs.get('location')
        description = inputs.get('description')
        start_time = datetime.strptime(inputs.get('start_time'), '%Y-%m-%dT%H:%M:%S')
        end_time = datetime.strptime(inputs.get('end_time'), '%Y-%m-%dT%H:%M:%S')
        timezone = inputs.get('timezone', 'UTC')
        return create_event(summary, location, description, start_time, end_time, timezone)
    except Exception as e:
        return f"An error occurred: {str(e)}"