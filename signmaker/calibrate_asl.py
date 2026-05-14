"""
calibrate_asl.py — запись жестов HELP, MEASURE, CONNECT
Совместим с mediapipe 0.10+, Python 3.14, Windows

Запуск: python calibrate_asl.py
"""

import sys
import pickle
import urllib.request
import time
import numpy as np
from pathlib import Path

# ── Проверка зависимостей ──────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("❌  pip install opencv-python"); sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌  pip install mediapipe"); sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("❌  pip install scikit-learn"); sys.exit(1)

# ── Настройки ──────────────────────────────────────────────────────────────────
GESTURES          = ["HELP", "MEASURE", "CONNECT"]
SAMPLES_PER_CLASS = 30
MODEL_PATH        = Path("asl_model.pkl")
TASK_PATH         = Path("hand_landmarker.task")
TASK_URL          = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Описание жестов (показывается на экране)
HINTS = {
    "HELP": [
        "HELP",
        "Правая рука: кулак, большой палец вверх",
        "Левая рука: открытая ладонь",
        "Кулак лежит на ладони",
        "Нужны ОБЕ руки в кадре!",
    ],
    "MEASURE": [
        "MEASURE",
        "Обе руки: указательные пальцы вытянуты",
        "Руки разведены в стороны",
        "Как показывают размер: <--- --->",
        "Нужны ОБЕ руки в кадре!",
    ],
    "CONNECT": [
        "CONNECT",
        "Обе руки: знак OK (большой + указательный в кольцо)",
        "Кольца двух рук сцеплены вместе",
        "Как звенья цепи",
        "Нужны ОБЕ руки в кадре!",
    ],
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def download_model():
    if TASK_PATH.exists():
        print(f"✓ hand_landmarker.task уже есть")
        return
    print("⬇  Скачиваю модель (~18 MB)...")
    urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    print("✓ Скачано!")

def features(hand_list):
    def normalize(hand_pts):
        # same normalization as server runtime:
        # translate to wrist and scale by max dist to wrist
        if not hand_pts or len(hand_pts) < 21:
            return [(0.0, 0.0, 0.0)] * 21
        x0, y0, z0 = float(hand_pts[0][0]), float(hand_pts[0][1]), float(hand_pts[0][2])
        dx = [float(p[0]) - x0 for p in hand_pts[:21]]
        dy = [float(p[1]) - y0 for p in hand_pts[:21]]
        dz = [float(p[2]) - z0 for p in hand_pts[:21]]
        scale = 0.0
        for i in range(21):
            d = (dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2) ** 0.5
            if d > scale:
                scale = d
        if scale < 1e-6:
            scale = 1.0
        return [(dx[i] / scale, dy[i] / scale, dz[i] / scale) for i in range(21)]

    out = []
    for i in range(2):
        if i < len(hand_list) and hand_list[i] and len(hand_list[i]) >= 21:
            for x, y, z in normalize(hand_list[i]):
                out += [x, y, z]
        else:
            out += [0.0] * 63
    return np.array(out, dtype=np.float32)

def draw_hands(frame, result, w, h):
    if not result.hand_landmarks:
        return
    for hand_lm in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (124, 106, 247), 2)
        for pt in pts:
            cv2.circle(frame, pt, 5, (80, 230, 160), -1)

def put_text(frame, text, y, scale=0.7, color=(255,255,255), bold=False):
    thickness = 2 if bold else 1
    cv2.putText(frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def main():
    download_model()

    # Инициализация детектора
    print("Инициализирую детектор рук...")
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_PATH)),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    # Открываем камеру
    print("Открываю камеру...")
    cap = None
    for cam_idx in range(3):
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"✓ Камера найдена (индекс {cam_idx})")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌  Камера не найдена! Проверь подключение веб-камеры.")
        sys.exit(1)

    # Устанавливаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ВАЖНО: создаём окно заранее и выносим на передний план
    WIN = "SignMaker — Калибровка жестов"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)

    # Прогрев камеры (первые кадры бывают чёрными)
    print("Прогрев камеры...")
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    print("\n" + "="*60)
    print("  ОКНО КАМЕРЫ ДОЛЖНО БЫТЬ ОТКРЫТО!")
    print("  Если не видно — нажми Alt+Tab или посмотри на панель задач")
    print("="*60)

    training_data = {g: [] for g in GESTURES}

    for g_idx, gesture in enumerate(GESTURES):
        hints = HINTS[gesture]
        count = 0
        need  = 2  # все три жеста двуручные

        # ── ЭКРАН 1: Инструкция перед жестом ──────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Жест {g_idx+1}/3: {gesture}")
        print(f"{'='*50}")
        for h in hints[1:]:
            print(f"  • {h}")
        print(f"\n  Прочитай, прими позу, нажми ЛЮБУЮ КЛАВИШУ в окне камеры")

        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            h_px, w_px = frame.shape[:2]

            # Затемняем фон
            overlay = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

            # Большой заголовок
            cv2.rectangle(frame, (0, 0), (w_px, 75), (20, 20, 50), -1)
            put_text(frame, f"ШАГ {g_idx+1}/3:  {gesture}",
                     50, scale=1.3, color=(80, 230, 160), bold=True)

            # Подсказки
            y = 110
            for line in hints[1:]:
                put_text(frame, line, y, scale=0.65, color=(220, 220, 220))
                y += 34

            # Стрелка вниз
            put_text(frame,
                     ">>> Прими позу и нажми ЛЮБУЮ КЛАВИШУ <<<",
                     h_px - 30, scale=0.75, color=(80, 230, 160), bold=True)

            cv2.imshow(WIN, frame)
            cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)  # поверх окон

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); sys.exit(0)
            if key != 255:
                waiting = False

        # ── ЭКРАН 2: Запись примеров ───────────────────────────────────────────
        print(f"\n  Запись! Нажимай ПРОБЕЛ {SAMPLES_PER_CLASS} раз, каждый раз немного меняй положение рук")

        flash_frames = 0  # мигание при успешном сохранении

        while count < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)
            h_px, w_px = frame.shape[:2]

            # Детекция рук
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            num    = len(result.hand_landmarks) if result.hand_landmarks else 0
            ok     = num >= need

            # Рисуем скелет рук
            draw_hands(frame, result, w_px, h_px)

            # Мигание зелёным при сохранении
            if flash_frames > 0:
                cv2.rectangle(frame, (0, 0), (w_px, h_px), (0, 200, 100), 6)
                flash_frames -= 1

            # Верхняя панель
            cv2.rectangle(frame, (0, 0), (w_px, 70), (10, 10, 30), -1)
            status_col = (80, 230, 160) if ok else (60, 80, 255)
            put_text(frame, f"{gesture}   {count}/{SAMPLES_PER_CLASS} сохранено",
                     45, scale=1.1, color=status_col, bold=True)

            # Статус рук — справа
            hand_txt = f"Рук: {num}/2"
            col_h = (80, 230, 160) if ok else (60, 80, 255)
            cv2.putText(frame, hand_txt,
                        (w_px - 160, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, col_h, 2, cv2.LINE_AA)

            if not ok:
                put_text(frame,
                         "!  Нужны ОБЕ руки в кадре!  !",
                         h_px // 2,
                         scale=0.95, color=(60, 80, 255), bold=True)

            # Прогресс-бар
            bar_y = h_px - 45
            cv2.rectangle(frame, (20, bar_y), (w_px - 20, bar_y + 12), (40, 40, 40), -1)
            filled = int((count / SAMPLES_PER_CLASS) * (w_px - 40))
            if filled > 0:
                cv2.rectangle(frame, (20, bar_y), (20 + filled, bar_y + 12), (80, 230, 160), -1)

            # Нижняя подсказка
            cv2.rectangle(frame, (0, h_px - 30), (w_px, h_px), (10, 10, 30), -1)
            put_text(frame,
                     "ПРОБЕЛ = сохранить    ENTER = перейти к след. жесту    Q = выход",
                     h_px - 8, scale=0.5, color=(160, 160, 160))

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); detector.close()
                print("\nВыход."); sys.exit(0)

            if key == 13:  # ENTER — пропустить
                print(f"  Пропуск. Записано: {count}")
                break

            if key == ord(' '):
                if result.hand_landmarks and ok:
                    hand_data = []
                    # стабилизируем порядок рук: сортируем по X запястья (слева-направо на изображении)
                    hands_sorted = sorted(
                        list(result.hand_landmarks)[:2],
                        key=lambda hnd: (hnd[0].x if len(hnd) > 0 else 0.0),
                    )
                    for hand_lm in hands_sorted:
                        hand_data.append([(lm.x, lm.y, lm.z) for lm in hand_lm])
                    training_data[gesture].append(hand_data)
                    count += 1
                    flash_frames = 4
                    print(f"  ✓  {count}/{SAMPLES_PER_CLASS}  — пример сохранён")
                else:
                    print(f"  ✗  Видно рук: {num} — нужно 2. Убедись что обе руки в кадре!")

        print(f"  Жест {gesture}: записано {count} примеров ✓")

    # ── Финальный экран ────────────────────────────────────────────────────────
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h_px, w_px = frame.shape[:2]
        overlay = np.zeros_like(frame)
        frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
        put_text(frame, "Все жесты записаны!", h_px//2 - 40,
                 scale=1.4, color=(80, 230, 160), bold=True)
        put_text(frame, "Обучаю модель... подожди несколько секунд",
                 h_px//2 + 20, scale=0.75, color=(220, 220, 220))
        cv2.imshow(WIN, frame)
        cv2.waitKey(1500)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # ── Обучение ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Обучаю Random Forest...")

    X, y = [], []
    for gesture, samples in training_data.items():
        for hand_list in samples:
            X.append(features(hand_list))
            y.append(gesture)

    total = len(X)
    if total < 6:
        print("❌  Слишком мало примеров. Запусти заново.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "gestures": GESTURES, "version": "v2"}, f)

    print(f"\n✅  ГОТОВО!")
    print(f"   Модель: {MODEL_PATH.absolute()}")
    print(f"   Жесты : {list(model.classes_)}")
    print(f"   Итого примеров: {total}")
    print("\n   Теперь запусти:  python app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
