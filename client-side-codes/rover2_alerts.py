import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging
import time


def send_rover2_alert(title, message):
    # Initialize Firebase with your service account key
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)

    # Your app's device token
    DEVICE_TOKEN = 'caau-AdQTseJxf6DRZpTn7:APA91bGrboDTMWiJSBVsaIX5axUxccU5RpPwBl70LNVIEs8fGOU8UR5szmcC56f_RXO7s-yONs3hNzD2tsRi621A3W915rldeE85ryrXSteyy-QJX-QvZ_c'
    try:
        message = messaging.Message(
            data={
                'roverId': '2',  # This identifies it as Rover 2
                'title': title,
                'message': message
            },
            token=DEVICE_TOKEN
        )
        
        response = messaging.send(message)
        print('Successfully sent Rover 2 alert')
        print('Response:', response)
        
    except Exception as e:
        print('Error sending Rover 2 alert:', str(e))

# Example usage:
if __name__ == "__main__":
    # You can call this function whenever Rover 2 needs to send an alert
    send_rover2_alert("alert", "alert")