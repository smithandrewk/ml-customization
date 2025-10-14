import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_discord_notification(message, webhook_url=None):
    """
    Send a notification to Discord using a webhook.

    Args:
        message: The message to send
        webhook_url: Discord webhook URL (optional, defaults to DISCORD_WEBHOOK_URL env variable)
    """
    webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')

    if not webhook_url:
        print("Error: No webhook URL provided")
        return False

    data = {"content": message}

    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Notification sent successfully!")
            return True
        else:
            print(f"Failed to send notification: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending notification: {e}")
        return False