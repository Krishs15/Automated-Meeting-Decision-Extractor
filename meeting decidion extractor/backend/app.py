"""
Flask API + web UI for the Meeting Decision Extraction System.
"""

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from extractor import extract_decisions
from train_model import MODEL_PATH, train_and_save

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.before_request
def ensure_model():
    if not MODEL_PATH.exists() and request.endpoint != "health":
        train_and_save()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})


@app.post("/api/extract")
def extract():
    payload = request.get_json(silent=True) or {}
    transcript = (payload.get("transcript") or "").strip()

    if not transcript:
        return jsonify({"error": "Please provide a meeting transcript."}), 400

    try:
        decisions = extract_decisions(transcript)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "decisions": decisions,
            "count": len(decisions),
        }
    )


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        train_and_save()
    app.run(debug=True, port=5000)
