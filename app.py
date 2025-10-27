import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import time # Import time (though less critical now)

# ------------------ Setup ------------------
st.set_page_config(page_title="Facial Recognition Attendance", layout="wide")
st.title("🧑‍💻 Facial Recognition Attendance System (Single-Shot Camera)")

# Define paths
FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"

# Ensure directories/files exist
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Haar Cascade face detector (optional, DeepFace has internal detection but this is fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ------------------ Helper Functions ------------------
def get_known_faces():
    """Returns a list of paths to known face images."""
    return [os.path.join(FACES_DIR, f) for f in os.listdir(FACES_DIR) if f.lower().endswith((".jpg",".png"))]


def recognize_face(img_path):
    """Recognizes a face image at img_path against the known faces database."""
    known_faces = get_known_faces()
    if known_faces:
        try:
            # Use VGG-Face for recognition
            result = DeepFace.find(img_path=img_path, db_path=FACES_DIR, enforce_detection=False, model_name="VGG-Face")
            
            if len(result[0]) > 0:
                # Extract the name from the path of the closest match
                matched_name = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
                return matched_name, False # Return name and is_new=False
        except Exception:
            # Ignore DeepFace errors (e.g., face not detected by DeepFace or no match)
            pass 
            
    return None, True # Return None and is_new=True (if no match or error)


def mark_attendance(name):
    """Marks attendance for the given name for today if not already marked. Returns True if marked, False otherwise."""
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Ensure the 'Date' column is standardized
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Check if this person has already signed in today
    if not ((df['Name'] == name) & (df['Date'] == today)).any():
        new_row = {"Name": name, "Date": today, "Time": datetime.now().strftime("%H:%M:%S")}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.toast(f"✅ Attendance marked for {name}!")
        return True # Attendance was successfully marked
    return False # Attendance was NOT marked


def register_new_face(img, name):
    """Saves the cropped face image with the given name."""
    filename = f"{name}.jpg"
    path = os.path.join(FACES_DIR, filename)
    # Convert RGB (Streamlit image format) back to BGR (OpenCV format) before saving
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def process_captured_image(file_bytes):
    """Handles the face detection, recognition, and attendance flow for a single captured image."""
    
    # 1. Convert Streamlit uploaded file bytes to OpenCV image
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR image
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB image for session state

    # 2. Detect faces using Haar Cascade (fast initial check)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        st.error("❌ No face detected in the captured image. Please ensure your face is clearly visible and try again.")
        return

    # Process the first detected face
    x, y, w, h = faces[0]
    
    # 3. Prepare image for DeepFace (temporary file is required)
    face_img_bgr = frame[y:y+h, x:x+w]
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, face_img_bgr)

    # 4. Recognize face
    name, is_new = recognize_face(temp_path)
    
    # 5. Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 6. Handle results and update state
    if is_new:
        # New Face Detected - Trigger stop and store info
        st.session_state.new_face_detected = True
        st.session_state.new_face_image = rgb_img[y:y+h, x:x+w]
        st.error("🚨 Unknown face detected! Proceed to registration.")
        
    elif name:
        # Known Face Detected - Check attendance
        mark_attendance(name)
        st.success(f"Welcome back, {name}! See attendance table for today's entry.")
    
    # Rerun to clear the camera input and reflect the new state (registration form/table update)
    st.rerun()


# ------------------ Session State ------------------
if "new_face_image" not in st.session_state:
    st.session_state.new_face_image = None
if "new_face_detected" not in st.session_state:
    st.session_state.new_face_detected = False

# ------------------ Main Application Layout ------------------

# --- Camera Input ---
st.subheader("▶️ Capture Face Image for Attendance")

# Use st.camera_input - the deployable component
captured_file = st.camera_input(
    "Click the 'Take Photo' button below to check attendance.",
    key="camera_capture",
    disabled=st.session_state.new_face_detected # Disable while registration is pending
)

if captured_file is not None:
    # We use a button to explicitly trigger processing, preventing it from running every frame
    if st.button("Process Captured Image", key="process_button"):
        process_captured_image(captured_file.read())


# ------------------ Register New Face ------------------
if st.session_state.new_face_detected:
    st.subheader("🆕 New Face Detected! Register Below")
    st.image(st.session_state.new_face_image, caption="Detected Face (Cropped)", width=300)
    
    new_name = st.text_input("Enter Name for New Face", key="new_name_input")
    
    if new_name and st.button("Register Face & Mark Attendance", key="register_button"):
        # Check if name already exists
        if f"{new_name}.jpg" in os.listdir(FACES_DIR):
             st.warning(f"A person named '{new_name}' already exists. Please choose a unique name or check the 'faces' directory.")
        else:
            register_new_face(st.session_state.new_face_image, new_name)
            mark_attendance(new_name)
            st.success(f"✅ New face registered and attendance marked for {new_name}")
            
            # Reset state and rerun to go back to the starting point
            st.session_state.new_face_detected = False
            st.session_state.new_face_image = None
            st.rerun() 


# ------------------ Attendance Table ------------------
st.subheader("📊 Attendance Records")
df = pd.read_csv(ATTENDANCE_FILE)
# Display the DataFrame
st.dataframe(df[df['Name'].notna()], use_container_width=True)


st.download_button(
    "Download Attendance CSV",
    data=open(ATTENDANCE_FILE,"rb"),
    file_name="attendance.csv",
    mime="text/csv"
)


# ------------------ Instructions ------------------
with st.expander("ℹ️ How to Use (Refactored Version)"):
    st.markdown("""
    1. Click the **'Take Photo'** button using the camera widget above. This uses *your* local webcam.
    2. Once the image is captured, click the **'Process Captured Image'** button.
    3. **Known Face:** If your face is recognized, attendance is marked for the day.
    4. **Unknown Face:** If a face is unknown, the camera input clears, and the **New Face Registration** form appears.
    """)