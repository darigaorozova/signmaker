import os, json, logging, pickle, time, socket
from pathlib import Path
from collections import deque, Counter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn

# ── Gemma 4 setup (по ТЗ) ──────────────────────────────────────────────────
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("signmaker")

try:
    # Optional: load .env if present
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# ── Загрузка модели ────────────────────────────────────────────────────────
MODEL_PATH = Path("asl_model.pkl")
clf = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        raw = pickle.load(f)
    clf = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

def _extract_hands(lm_data):
    """
    Accepts:
      - {"hand_0":[{x,y,z}...], "hand_1":[...]}
      - {"0":[...], "1":[...]} or {0:[...],1:[...]}
      - [[(x,y,z)*21], [(x,y,z)*21]] (list-of-hands)
    Returns a list with up to 2 hands, each = list of 21 points.
    """
    if lm_data is None:
        return []

    if isinstance(lm_data, list):
        # list-of-hands (each hand is list of points)
        return lm_data[:2]

    if isinstance(lm_data, dict):
        hands = []
        for i in range(2):
            pts = lm_data.get(f"hand_{i}") or lm_data.get(str(i)) or lm_data.get(i)
            if pts:
                hands.append(pts)
        return hands

    return []


def _normalize_hand(hand_pts):
    """
    Normalization that makes the classifier far less sensitive to camera distance/position:
    - translate so wrist (landmark 0) is at origin
    - scale by max distance to wrist (avoid division by 0)
    Works for both dict points and tuple/list points.
    """
    if not hand_pts or len(hand_pts) < 21:
        return [(0.0, 0.0, 0.0)] * 21

    def get_xyz(p):
        if isinstance(p, dict):
            return float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))
        return float(p[0]), float(p[1]), float(p[2])

    xs = [get_xyz(p)[0] for p in hand_pts[:21]]
    ys = [get_xyz(p)[1] for p in hand_pts[:21]]
    zs = [get_xyz(p)[2] for p in hand_pts[:21]]

    x0, y0, z0 = xs[0], ys[0], zs[0]
    dx = [x - x0 for x in xs]
    dy = [y - y0 for y in ys]
    dz = [z - z0 for z in zs]

    scale = 0.0
    for i in range(21):
        d = (dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2) ** 0.5
        if d > scale:
            scale = d
    if scale < 1e-6:
        scale = 1.0

    return [(dx[i] / scale, dy[i] / scale, dz[i] / scale) for i in range(21)]


def landmarks_to_features(lm_data) -> np.ndarray:
    hands = _extract_hands(lm_data)
    out = []
    for i in range(2):
        if i < len(hands) and hands[i] and len(hands[i]) >= 21:
            norm = _normalize_hand(hands[i])
            for x, y, z in norm:
                out += [x, y, z]
        else:
            out += [0.0] * 63
    return np.array(out, dtype=np.float32)


def _safe_predict(model, feat: np.ndarray) -> tuple[str, float]:
    if model is None:
        return "UNKNOWN", 0.0
    pred = model.predict(feat)[0]
    try:
        conf = float(np.max(model.predict_proba(feat)[0]))
    except Exception:
        conf = 1.0
    gesture = str(pred).strip().upper()
    gesture = gesture.replace("NP.STR_('", "").replace("')", "")
    return gesture, conf


def _placeholder_svg(title: str, icon: str) -> str:
    # Simple inline SVG so frontend can render a "card image" without needing real PNG files.
    t = (title or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    i = (icon or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7c6af7"/>
      <stop offset="1" stop-color="#4ecca3"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" rx="28" fill="#131624"/>
  <rect x="22" y="22" width="596" height="436" rx="22" fill="url(#g)" opacity="0.18"/>
  <text x="60" y="130" font-size="92" font-family="Segoe UI, Arial">{i}</text>
  <text x="60" y="220" font-size="34" font-weight="800" fill="#eef0ff" font-family="Segoe UI, Arial">{t}</text>
  <text x="60" y="270" font-size="22" fill="#7a7f9a" font-family="Segoe UI, Arial">Test image (replace with PNG later)</text>
</svg>"""

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't need to be reachable; no traffic is sent
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _is_port_free(port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.close()
        return True
    except OSError:
        return False


def _pick_port(preferred: int) -> int:
    # Choose first free port in [preferred..preferred+20]
    for p in range(preferred, preferred + 21):
        if _is_port_free(p):
            return p
    return preferred


def _print_banner(port: int) -> None:
    ip = _get_local_ip()
    gemma = "– no key (set GEMMA_API_KEY)" if not (GEMMA_API_KEY or "").strip() else "✓ key set"
    if not (GEMMA_API_KEY or "").strip():
        logger.info("i No GEMMA_API_KEY set — running without AI text generation\n")

    line = "─" * 70
    logger.info(line)
    logger.info("\n✋ SignMaker MVP\n")
    logger.info(line + "\n")
    logger.info(f"💻 http://localhost:{port}\n")
    logger.info(f"📱 http://{ip}:{port} (same Wi‑Fi)\n")
    logger.info(f"Gemma-4 : {gemma}\n")
    logger.info("Ctrl+C to stop\n")
    logger.info(line + "\n")

# ── Контент строго по ТЗ ──────────────────────────────────────────────────
APRON_STEPS = {
    1: {
        "title": "Замеры",
        "icon": "📏",
        "cards": [
            {
                "title": "Обхват талии",
                "text": "Сначала измерь обхват талии.",
                "image": "/static/images/waist.png"
            },
            {
                "title": "Обхват бёдер",
                "text": "Затем измерь обхват бёдер.",
                "image": "/static/images/hips.png"
            },
            {
                "title": "Длина подола",
                "text": "Отмерь длину от талии до колен (примерно 35–40 см).",
                "image": "/static/images/length.png"
            },
            {
                "title": "Высота грудки",
                "text": "Измерь расстояние от талии до верха груди.",
                "image": "/static/images/chest.png"
            }
        ]
    },
    6: {
        "title": "Сборка",
        "icon": "🔗",
        "cards": [
            {
                "title": "1. Пришиваем грудку",
                "text": "Пришей верхнюю часть фартука (грудку) к самому центру пояса.",
                "image": "/static/images/attach_top.png"
            },
            {
                "title": "2. Соединяем подол",
                "text": "Теперь пришей нижнюю часть (подол) к нижней стороне пояса.",
                "image": "/static/images/attach_bottom.png"
            },
            {
                "title": "3. Добавляем завязку",
                "text": "Прикрепи шейную лямку к верхним углам грудки.",
                "image": "/static/images/add_strap.png"
            },
            {
                "title": "4. Готово!",
                "text": "Ура! Все детали соединены. Твой фартук почти готов!",
                "image": "/static/images/done_apron.png"
            }
        ]
    }
}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))

@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    
    # Состояние системы
    ai_active = False 
    last_gesture = None
    last_gesture_at = 0.0
    pending_step_prompt = None  # 1 or 6
    last_rx_log_at = 0.0
    # rolling window to make RandomForest "probabilities" usable
    # (RF often outputs ~0.35–0.55 even when it's correct)
    win = deque(maxlen=10)  # items: (gesture, conf)
    
    try:
        # Стартовое сообщение
        await ws.send_json({"text": "👋 <b>SignMaker готов!</b>", "stage_name": "Ready", "stage_icon": "🎯"})
        
        while True:
            data = await ws.receive_json()
            
            msg_type = data.get("type")
            now_ts = time.time()
            if (now_ts - last_rx_log_at) > 2.0:
                try:
                    keys = ",".join(sorted([str(k) for k in data.keys()]))
                except Exception:
                    keys = "?"
                print(f"DEBUG: rx type={msg_type} keys={keys}", flush=True)
                last_rx_log_at = now_ts

            # 1) ОСНОВНОЙ ПУТЬ: landmarks -> твоя обученная модель (HELP/MEASURE/CONNECT)
            if msg_type == "landmarks":
                if clf is None:
                    continue

                feat = landmarks_to_features(data.get("landmarks")).reshape(1, -1)
                gesture, conf = _safe_predict(clf, feat)

                # Debug: видим, что реально предсказывает модель в рантайме
                print(
                    f"DEBUG: gesture={gesture} conf={conf:.2f} active={ai_active} hands={data.get('hands')}",
                    flush=True,
                )

                # Полностью игнорируем любые THUMB* (убираем ложные срабатывания)
                if "THUMB" in gesture:
                    continue

                hands = int(data.get("hands") or 0)
                # HELP may be one-handed now (Thumb_Up), but MEASURE/CONNECT are two-handed.
                if gesture == "HELP" and hands >= 1:
                    win.append((gesture, conf))
                elif hands >= 2:
                    win.append((gesture, conf))

                stable_gesture = None
                stable_conf = 0.0
                if len(win) >= 6:
                    counts = Counter([g for g, _ in win])
                    g0, n0 = counts.most_common(1)[0]
                    # Require dominance in the last frames (less strict so it actually triggers)
                    if n0 >= 7:
                        stable_gesture = g0
                        stable_conf = float(np.mean([c for g, c in win if g == g0]))

                now = time.time()
                # антидребезг: не реагировать чаще чем раз в ~1.2с на одну и ту же команду
                if stable_gesture is None:
                    continue

                if stable_gesture == last_gesture and (now - last_gesture_at) < 1.2:
                    continue

                # VICTORY: stage completion / positive reinforcement
                if stable_gesture in ("VICTORY", "Victory") :
                    await ws.send_json({
                        "text": "🌟 <b>Блестящая работа!</b> Ты настоящий мастер. Теперь мы готовы двигаться дальше!",
                        "stage_name": "Успех!",
                        "stage_icon": "🎉",
                        "show_input": False,
                    })
                    last_gesture = "VICTORY"
                    last_gesture_at = now
                    win.clear()
                    continue

                # HELP: greet + (re)start flow (also works for a new child)
                if stable_gesture == "HELP":
                    ai_active = True
                    pending_step_prompt = None
                    await ws.send_json({
                        "text": "👋 <b>Привет! Я твой ИИ-помощник.</b> С чем тебе помочь? Покажи жест 'Measure' (Замеры) или 'Connect' (Сборка), и начнем!",
                        "stage_name": "Выбор этапа",
                        "stage_icon": "👋",
                        "show_input": False,
                        "hide_cards": True,
                    })
                    last_gesture = stable_gesture
                    last_gesture_at = now
                    win.clear()
                    continue

                # If not active — ignore other gestures
                if not ai_active:
                    last_gesture = stable_gesture
                    last_gesture_at = now
                    win.clear()
                    continue

                # MEASURE / CONNECT: ask for manual confirmation (Step 1 / Step 6)
                if stable_gesture == "MEASURE":
                    pending_step_prompt = 1
                    await ws.send_json({
                        "show_input": True,
                        "text": "📏 <b>Хочешь узнать, как делать замеры?</b> Нажми «1» на клавиатуре для подтверждения.",
                        "stage_name": "Clarification", "stage_icon": "❓",
                    })
                    last_gesture = stable_gesture
                    last_gesture_at = now
                    win.clear()
                    continue

                if stable_gesture == "CONNECT":
                    pending_step_prompt = 6
                    await ws.send_json({
                        "show_input": True,
                        "text": "🔗 <b>Переходим к сборке фартука?</b> Введи «6» на клавиатуре, чтобы открыть инструкции.",
                        "stage_name": "Clarification", "stage_icon": "❓",
                    })
                    last_gesture = stable_gesture
                    last_gesture_at = now
                    win.clear()
                    continue

            # 1b) Совместимость: если фронт по-прежнему шлёт только type:"gesture"
            elif msg_type == "gesture":
                g = str(data.get("gesture", "")).strip()
                # Treat Thumb_Up as HELP (requested), even if custom model doesn't fire
                if g == "Thumb_Up":
                    g = "HELP"
                if g == "Victory":
                    g = "VICTORY"
                # Можно вывести на экран, что сервер получил, чтобы дебажить протокол
                # (логи в терминале)
                print(f"DEBUG: mp_gesture={g} conf={data.get('confidence')} hands={data.get('hands')}", flush=True)

                # VICTORY via frontend gesture event
                if g == "VICTORY":
                    win.clear()
                    await ws.send_json({
                        "text": "🌟 <b>Блестящая работа!</b> Ты настоящий мастер. Теперь мы готовы двигаться дальше!",
                        "stage_name": "Успех!",
                        "stage_icon": "🎉",
                        "show_input": False,
                    })
                    continue

                # HELP via frontend gesture event (always re-greet + reset flow)
                if g in ("HELP", "Open_Palm"):
                    ai_active = True
                    pending_step_prompt = None
                    win.clear()
                    await ws.send_json({
                        "text": "👋 <b>Привет! Я твой ИИ-помощник.</b> С чем тебе помочь? Покажи жест 'Measure' (Замеры) или 'Connect' (Сборка), и начнем!",
                        "stage_name": "Выбор этапа",
                        "stage_icon": "👋",
                        "show_input": False,
                        "hide_cards": True,
                    })
                    continue

            elif msg_type == "hello":
                # lightweight debug reply so you can see backend is receiving messages
                await ws.send_json({
                    "text": "✅ Связь с сервером есть. Покажи <b>HELP</b> (двумя руками).",
                    "stage_name": "Ready",
                    "stage_icon": "🎯",
                })

            # 2. ОБРАБОТКА ВВОДА С КЛАВИАТУРЫ (Confirmation)
            elif msg_type == "step":
                try:
                    step_num = int(data.get("step", 0))
                    if step_num in APRON_STEPS:
                        step_data = APRON_STEPS[step_num]
                        cards = []
                        for c in step_data["cards"]:
                            cards.append({
                                "title": c["title"],
                                "text": c["text"],
                                "image": c.get("image"),
                            })
                        await ws.send_json({
                            "text": f"✅ <b>Загрузка карточек для Шага {step_num}...</b>",
                            "stage_name": step_data["title"],
                            "stage_icon": step_data["icon"],
                            "cards": cards,
                            "show_input": False,
                        })
                except:
                    pass

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    preferred = int(os.getenv("PORT", "7860"))
    port = _pick_port(preferred)
    _print_banner(port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)