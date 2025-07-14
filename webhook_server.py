from fastapi import FastAPI, Request
import uvicorn
import datetime
import os
import sys

# Ensure main.py is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import predict_ticket, get_requester_info, get_group_id_by_name, FRESHSERVICE_API_KEY, FRESHSERVICE_DOMAIN

import requests

app = FastAPI()

@app.post("/webhook/freshservice")
async def receive_webhook(request: Request):
    data = await request.json()
    ticket_id = data.get("ticket_id") or data.get("id")
    subject = data.get("subject", "")
    description = data.get("description", "")
    requester_id = data.get("requester_id")

    # Fetch requester info if available
    requester_info = get_requester_info(requester_id) if requester_id else {"email": "", "location": "", "department_name": ""}

    # Predict group using your ML model
    predicted_group, confidence_score = predict_ticket(
        description=description,
        subject=subject,
        requester_id=requester_id
    )
    predicted_group_name = predicted_group.get("Group", "")
    confidence_percent = f"{int(round(confidence_score.get('Group', 0) * 100))}%"

    # Optionally, auto-assign group via Freshservice API
    group_id = None
    api_status = None
    api_response = None
    if predicted_group_name:
        try:
            group_id = get_group_id_by_name(predicted_group_name)
            if group_id:
                update_url = f"{FRESHSERVICE_DOMAIN}/api/v2/tickets/{ticket_id}"
                headers = {"Content-Type": "application/json"}
                payload = {"group_id": int(group_id)}
                response = requests.put(
                    update_url,
                    auth=(FRESHSERVICE_API_KEY, "X"),
                    headers=headers,
                    json=payload,
                    verify=False
                )
                api_status = response.status_code
                api_response = response.text
        except Exception as e:
            api_status = "ERROR"
            api_response = str(e)

    # Log the result
    print(f"[{datetime.datetime.now()}] Webhook received for Ticket {ticket_id}:")
    print(f"  Subject: {subject}")
    print(f"  Predicted Group: {predicted_group_name} (Confidence: {confidence_percent})")
    print(f"  Assigned Group ID: {group_id}, API Status: {api_status}")

    return {
        "status": "received",
        "ticket_id": ticket_id,
        "predicted_group": predicted_group_name,
        "confidence": confidence_percent,
        "assigned_group_id": group_id,
        "api_status": api_status,
        "api_response": api_response
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)