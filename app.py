import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ------------------ Setup ------------------
st.set_page_config(page_title="Facial Recognition Attendance", layout="wide")
st.title("🧑‍💻 Facial Recognition Attendance System (Webcam Only)")

FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"

# Ensure directories/files exist
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------ Helper Functions ------------------
def get_known_faces():
    return [os.path.join(FACES_DIR, f) for f in os.listdir(FACES_DIR) if f.lower().endswith((".jpg",".png"))]

def recognize_face(img):
    known_faces = get_known_faces()
    if known_faces:
        try:
            result = DeepFace.find(img_path=img, db_path=FACES_DIR, enforce_detection=False)
            if len(result[0]) > 0:
                matched_name = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
                return matched_name, False
        except:
            pass
    return None, True

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    if not ((df['Name'] == name) & (df['Date'] == today)).any():
        new_row = {"Name": name, "Date": today, "Time": datetime.now().strftime("%H:%M:%S")}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

def register_new_face(img, name):
    filename = f"{name}.jpg"
    path = os.path.join(FACES_DIR, filename)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ------------------ Session State ------------------
if "new_face_image" not in st.session_state:
    st.session_state.new_face_image = None
if "new_face_detected" not in st.session_state:
    st.session_state.new_face_detected = False
if "pause_webcam" not in st.session_state:
    st.session_state.pause_webcam = False

# ------------------ Webcam Transformer ------------------
class FaceAttendance(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        if self.frame_count % 10 == 0 and not st.session_state.pause_webcam:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = rgb_img[y:y+h, x:x+w]
                name, is_new = recognize_face(face_img)
                if is_new and not st.session_state.new_face_detected:
                    st.session_state.new_face_detected = True
                    st.session_state.new_face_image = face_img
                    st.session_state.pause_webcam = True
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
                    cv2.putText(img, "New Face Detected! Pausing...", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                elif name:
                    mark_attendance(name)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if st.session_state.pause_webcam:
            # Show black frame while waiting for registration
            img = np.zeros((720,1280,3), dtype=np.uint8)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ Start Webcam ------------------
st.warning("Starting webcam. Allow camera access in your browser.")

webrtc_streamer(
    key="attendance",
    video_transformer_factory=FaceAttendance,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
)

# ------------------ Register New Face ------------------
if st.session_state.new_face_detected:
    st.subheader("🆕 New Face Detected! Register Below")
    st.image(st.session_state.new_face_image, caption="Detected Face", width=300)
    new_name = st.text_input("Enter Name for New Face")
    if new_name and st.button("Register Face"):
        register_new_face(st.session_state.new_face_image, new_name)
        mark_attendance(new_name)
        st.success(f"✅ New face registered and attendance marked for {new_name}")
        st.session_state.new_face_detected = False
        st.session_state.new_face_image = None
        st.session_state.pause_webcam = False  # Resume webcam

# ------------------ Attendance Table ------------------
st.subheader("📊 Attendance Records")
df = pd.read_csv(ATTENDANCE_FILE)
st.dataframe(df)

st.download_button(
    "Download Attendance CSV",
    data=open(ATTENDANCE_FILE,"rb"),
    file_name="attendance.csv",
    mime="text/csv"
)

# ------------------ Instructions ------------------
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Webcam scans for faces continuously.
    2. When a new face is detected, webcam pauses and shows the detected face above.
    3. Enter the person's name and click 'Register Face'.
    4. Attendance is automatically marked for known faces.
    5. Webcam resumes automatically after registration.
    6. Ensure your face is well-lit and clearly visible.
    """)
