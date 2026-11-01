# CARESYNQ — simple hackathon prototype

A lightweight website using HTML, CSS, JavaScript, Python Flask and SQLite.

## 1. Requirements
- Python 3.10 or newer
- VS Code (recommended)

## 2. Open the project
Extract the ZIP, open the `CARESYNQ` folder in VS Code, then open Terminal.

## 3. Create a virtual environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

## 5. Run the website
```bash
cd backend
python app.py
```
Open http://127.0.0.1:5000 in Chrome.

## Demo doctor account
- Email: `doctor@caresynq.demo`
- Password: `12345678`

Register a mother account yourself to test the mother dashboard, activity tracker and wellbeing check-in.

## Notes
- SQLite database `caresynq.db` is created automatically when the Flask app starts.
- All mother accounts created through registration have the `mother` role. The doctor account is seeded for local demonstration.
- This is a hackathon prototype, not a production health application. Do not enter real patient data.
- The activity rule in `backend/model.py` is a transparent demonstration rule, not a validated ML model and not a postpartum-depression risk score.
- Before deployment: set a strong secret key through environment configuration, disable Flask debug mode, add CSRF protection, HTTPS, rate limiting, access auditing, secure database backups and appropriate consent/privacy governance. Clinical use requires qualified clinical oversight and validation.
