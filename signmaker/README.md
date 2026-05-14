# ✋ SignMaker MVP

ASL gesture recognition + apron sewing step-by-step guide.  
Gestures are detected **in the browser** via MediaPipe (no Python ML required to start).

---

## Quick Start (5 minutes)

### 1 — Install Python dependencies

```bash
pip install fastapi uvicorn python-dotenv google-genai
```

> The heavier packages (`mediapipe`, `opencv-python`, `scikit-learn`) are only
> needed if you want to train custom gestures. Skip them for now.

### 2 — (Optional) Set up Gemma API key

Copy `.env.example` → `.env` and paste your key:

```
GEMMA_API_KEY=AIza...your_key_here...
```

**How to get a free key:**
1. Go to → <https://aistudio.google.com/apikey>
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key (starts with `AIza`)
5. Paste it into `.env`

Without a key the app still works — it just won't generate AI commentary.

### 3 — Run the server

```bash
python app.py
```

### 4 — Open in browser

| Device | URL |
|---|---|
| This computer | <http://localhost:7860> |
| Phone / iPad (same Wi-Fi) | `http://<your-local-ip>:7860` (printed in terminal) |

Allow camera access when the browser asks.

---

## How it works

```
Browser (MediaPipe JS)
  → detects hand landmarks in real-time (30 fps)
  → recognises gesture (Open_Palm = HELP, etc.)
  → holds gesture for ~1 second to confirm
  → sends {type:"gesture", gesture, confidence, hands} via WebSocket

Python (FastAPI)
  → maps gesture → step number
  → asks Gemma for a short encouraging sentence (optional)
  → sends back instruction cards as SVG + text

Child types step number → server sends full instruction cards
```

---

## Gesture → Step mapping

| Gesture (MediaPipe built-in) | Step |
|---|---|
| `Open_Palm` (both hands) | HELP — shows input box |
| `Pointing_Up` | Step 1 — Measurements |
| `Closed_Fist` | Step 5 — Sewing |
| `Victory` ✌️ | Step 7 — Quality Check |
| `ILoveYou` 🤟 | Step 6 — Connecting Parts |
| `Thumb_Up` 👍 | Step 8 — Done |

Custom gestures (MEASURE, CONNECT, etc.) require running `calibrate_asl.py` first.

---

## Custom gesture calibration (optional)

If you want the camera to recognise the exact gestures from the Tech Spec
(MEASURE, CONNECT, SEW…):

```bash
# Extra dependencies
pip install mediapipe opencv-python scikit-learn

# Record your gestures (follow on-screen instructions)
python calibrate_asl.py

# Then start the server as usual
python app.py
```

`app.py` automatically loads `asl_model.pkl` if it exists.

---

## File structure

```
signmaker/
├── app.py              ← FastAPI server (main entry point)
├── index.html          ← Frontend (MediaPipe JS + WebSocket UI)
├── asl_classifier.py   ← Helper: train / load custom gesture model
├── calibrate_asl.py    ← Tool: record gesture samples via webcam
├── requirements.txt    ← All Python dependencies
├── .env.example        ← Copy to .env and add your API key
└── README.md
```
