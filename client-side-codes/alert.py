import requests

def send_alert():
    API_KEY = "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"  # Replace with actual API key from Brevo
    SENDER_EMAIL = "kilopilo998@gmail.com"  # Must be your Brevo account email
    TO_EMAIL = "parvativiswanathan@iisertvm.ac.in"  # Change to recipient email

    url = "https://api.brevo.com/v3/smtp/email"

    payload = {
        "sender": {"name": "Rover Alert", "email": SENDER_EMAIL},
        "to": [{"email": TO_EMAIL, "name": "Farmer"}],
        "subject": "Rover Alert Notification",
        "htmlContent": "<h3>Warning! The rover detected an immediate irrigation requirement in your rice field.</h3>"
    }

    headers = {
        "accept": "application/json",
        "api-key": API_KEY,
        "content-type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print("✅ Email sent successfully!")
    else:
        print(f"❌ Error sending email: {response.text}")

