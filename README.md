import cv2
import time
import subprocess
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# MEDIAPIPE HAND TRACKING
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ==========================================
# KAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera konnte nicht geöffnet werden!")
    exit()

print()
print("======================================")
print("🖐️  HAND DESKTOP CONTROL")
print("======================================")
print("⬅️  Links wischen  = Desktop wechseln")
print("➡️  Rechts wischen = Desktop wechseln")
print("Q  = Beenden")
print("======================================")
print()


# ==========================================
# SWIPE EINSTELLUNGEN
# ==========================================

start_x = None
start_time = None

SWIPE_DISTANCE = 150
MAX_SWIPE_TIME = 1.0
COOLDOWN = 1.2

last_swipe = 0


# ==========================================
# DESKTOP WECHSELN
# ==========================================

def desktop_left():

    print("⬅️ LINKS WISCH!")

    subprocess.run([
        "ydotool",
        "key",
        "29:1",
        "56:1",
        "105:1",
        "105:0",
        "56:0",
        "29:0"
    ])


def desktop_right():

    print("➡️ RECHTS WISCH!")

    subprocess.run([
        "ydotool",
        "key",
        "29:1",
        "56:1",
        "106:1",
        "106:0",
        "56:0",
        "29:0"
    ])


# ==========================================
# HAUPTSCHLEIFE
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Kamera-Frame konnte nicht gelesen werden.")
        continue

    # Spiegelverkehrt anzeigen
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Bild
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Hand erkennen
    result = detector.detect(mp_image)

    now = time.time()


    # ======================================
    # HAND GEFUNDEN
    # ======================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Wrist = Landmark 0
        wrist = hand[0]

        h, w, _ = frame.shape

        x = int(wrist.x * w)
        y = int(wrist.y * h)

        # Punkt auf Hand anzeigen
        cv2.circle(
            frame,
            (x, y),
            12,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            "HAND ERKANNT",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==================================
        # STARTPOSITION SPEICHERN
        # ==================================

        if start_x is None:

            start_x = x
            start_time = now


        # Bewegung berechnen
        distance = x - start_x
        elapsed = now - start_time


        cv2.putText(
            frame,
            f"Bewegung: {distance}px",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ==================================
        # SWIPE ERKENNEN
        # ==================================

        if (
            elapsed < MAX_SWIPE_TIME
            and abs(distance) > SWIPE_DISTANCE
            and now - last_swipe > COOLDOWN
        ):

            # Nach LINKS
            if distance < 0:

                desktop_left()


            # Nach RECHTS
            else:

                desktop_right()


            # Cooldown starten
            last_swipe = now

            # Neue Bewegung vorbereiten
            start_x = None
            start_time = None


    # ======================================
    # KEINE HAND
    # ======================================

    else:

        start_x = None
        start_time = None


    # ======================================
    # KAMERA-FENSTER
    # ======================================

    cv2.imshow(
        "Hand Desktop Control",
        frame
    )


    # Q = Beenden
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# AUFRÄUMEN
# ==========================================

cap.release()

detector.close()

cv2.destroyAllWindows()
