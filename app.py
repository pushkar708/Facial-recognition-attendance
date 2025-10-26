import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import time # Import time for a small pause before clearing the frame

# ------------------ Setup ------------------
st.set_page_config(page_title="Facial Recognition Attendance", layout="wide")
st.title("🧑‍💻 Facial Recognition Attendance System (OpenCV Loop)")


FACES_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"


# Ensure directories/files exist
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
# Initialize or reset attendance file if empty
if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)


# Haar Cascade face detector
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
                matched_name = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
                return matched_name, False
        except Exception:
            pass # Ignore DeepFace errors for simplicity (e.g., face not detected by DeepFace)
    return None, True


def mark_attendance(name):
    """Marks attendance for the given name for today if not already marked. Returns True if marked, False otherwise."""
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # FIX 1: Ensure the 'Date' column is standardized to YYYY-MM-DD for comparison
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


# ------------------ Session State ------------------
if "new_face_image" not in st.session_state:
    st.session_state.new_face_image = None
if "new_face_detected" not in st.session_state:
    st.session_state.new_face_detected = False
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


# ------------------ Main Camera Loop ------------------
def run_camera_loop():
    """Initializes and runs the OpenCV camera feed loop."""
    
    st.session_state.camera_running = True
    
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        st.error("Error: Could not open camera. Please check camera permissions.")
        st.session_state.camera_running = False
        return

    frame_placeholder = st.empty()
    frame_count = 0
    
    # NEW: State variables to hold the drawing information across frames
    # Initialize with None
    current_draw_info = None 
    
    while cap.isOpened() and not st.session_state.new_face_detected and st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # --- A. Heavy Processing (DeepFace) - Only runs every 10th frame ---
        if frame_count % 3 == 0:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # === NEW LOGIC START ===
            if len(faces) == 0:
                # If no faces are detected by Haar Cascade:
                current_draw_info = None # Clear the box status
                frame_count = 0 # Reset frame count to run detection immediately when a face returns
            # === NEW LOGIC END ===
            # Reset draw info if no faces are detected in this heavy frame
            current_draw_info = None 
            
            for (x, y, w, h) in faces:
                # 1. Prepare image for DeepFace (temporary file is required)
                face_img_bgr = frame[y:y+h, x:x+w]
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_img_bgr)

                # 2. Recognize face
                name, is_new = recognize_face(temp_path)
                
                # 3. Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # 4. Handle results and store drawing info
                if is_new:
                    # New Face Detected - Trigger stop and store info
                    st.session_state.new_face_detected = True
                    st.session_state.new_face_image = rgb_img[y:y+h, x:x+w]
                    st.session_state.camera_running = False
                    current_draw_info = (x, y, w, h, "UNKNOWN! Stopping...", (0, 0, 255))
                    break 

                elif name:
                    # Known Face Detected - Check attendance
                    attendance_marked = mark_attendance(name) 
                    
                    if attendance_marked:
                        # Stop camera after first successful check-in
                        st.session_state.camera_running = False
                        current_draw_info = (x, y, w, h, f"{name} - Marked!", (0, 255, 0))
                        time.sleep(1) 
                        break 
                    else:
                        # Already marked today - continue scanning for others
                        current_draw_info = (x, y, w, h, f"{name} (Already Marked)", (100, 100, 100))
                        
            if not st.session_state.camera_running: 
                break 
        
        # --- B. Drawing Logic - Runs on EVERY frame ---
        if current_draw_info:
            # Use the last successfully detected coordinates and status for drawing
            x, y, w, h, text, color = current_draw_info
            
            # Draw the box using the stored data
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display the frame and update the container
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Attendance Scan")
        
    
    # Cleanup after loop breaks
    cap.release()
    cv2.destroyAllWindows()
    
    frame_placeholder.empty()

    if st.session_state.new_face_detected or not st.session_state.camera_running:
        st.rerun()
# ------------------ Start Camera Button ------------------
st.subheader("▶️ Start Attendance Scan")

if st.session_state.new_face_detected:
    st.error("🚨 **NEW FACE DETECTED!** Complete the registration below.")
elif st.session_state.camera_running:
    # Use a placeholder for the status message while running
    st.info("Scanning in progress...")
else:
    if st.button("Start/Resume Camera"):
        run_camera_loop()


# ------------------ Register New Face ------------------
if st.session_state.new_face_detected:
    st.subheader("🆕 New Face Detected! Register Below")
    st.image(st.session_state.new_face_image, caption="Detected Face", width=300)
    
    new_name = st.text_input("Enter Name for New Face", key="new_name_input")
    
    if new_name and st.button("Register Face", key="register_button"):
        register_new_face(st.session_state.new_face_image, new_name)
        mark_attendance(new_name)
        st.success(f"✅ New face registered and attendance marked for {new_name}")
        
        # Reset state and rerun to go back to the starting point
        st.session_state.new_face_detected = False
        st.session_state.new_face_image = None
        st.rerun() # FIX 1: Use st.rerun()


# ------------------ Attendance Table ------------------
st.subheader("📊 Attendance Records")
df = pd.read_csv(ATTENDANCE_FILE)
# Display the DataFrame, filtering out any rows where the name became 'None' due to previous logic issues
st.dataframe(df[df['Name'].notna()])


st.download_button(
    "Download Attendance CSV",
    data=open(ATTENDANCE_FILE,"rb"),
    file_name="attendance.csv",
    mime="text/csv"
)


# ------------------ Instructions ------------------
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Click the **'Start/Resume Camera'** button.
    2. The camera will start scanning for faces using an OpenCV loop.
    3. If a **known face** is detected, attendance is marked for the day, and the camera **stops automatically**.
    4. If an **unknown face** is detected, the camera loop **stops immediately**, the camera window is cleared, and the registration form appears.
    """)