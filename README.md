# Human Detection and Threat Alert System (BSF, MHA)

This project simulates an agentic AI–based border surveillance system. It uses MobileNet SSD to detect humans in real-time and sends alerts via Power Automate.

---

## Setup

- Clone the repository
- Create a virtual environment
- Install dependencies from `requirements.txt`
- Add a `.env` file in the project root with your Power Automate URL:

FLOW_URL=https://your-flow-url-here


- Run the system: python human_detector_sender.py

---

## How it works

The system captures video from webcam, detects humans, and sends a POST request to a Power Automate flow when a person is detected. A cooldown is applied to avoid repeated alerts.

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Power Automate
- MobileNet SSD

---

## Notes

- The `.env` file is excluded from Git to protect sensitive information.
- This setup is meant for local simulation; production deployment would require further optimizations.

---
