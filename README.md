# Facial Recognition Attendance System

This project is a **Facial Recognition-Based Attendance System** built using **Python** and the **DeepFace** library. It automates the process of marking attendance by recognizing faces through a webcam feed.

## 🚀 Features

- Real-time face detection and recognition  
- Automated attendance logging with timestamps  
- Supports multiple face recognition models via DeepFace  
- Stores attendance data in CSV or database format  
- Simple and modular Python codebase  

## 🧰 Tech Stack

- **Language:** Python 3.8+  
- **Libraries:** DeepFace, OpenCV, NumPy, Pandas  
- **Storage:** CSV or SQLite (optional)  

## 📦 Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/Facial-recognition-attendance.git
   cd Facial-recognition-attendance
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate    # For Linux/Mac
   venv\Scripts\activate       # For Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## 🧠 How It Works

1. The script captures live video using your webcam.  
2. Each frame is analyzed using DeepFace for facial recognition.  
3. If a match is found in the dataset, the user’s name and timestamp are logged.  
4. Attendance is automatically saved to a CSV file.

## ⚙️ Usage

Run the main Python script:
```
python main.py
```

Add known faces to the `images/` directory; each image filename should correspond to the person's name.

## 🖼 Example Output

```
[INFO] John_Doe recognized — attendance recorded at 09:12:34
[INFO] Jane_Smith recognized — attendance recorded at 09:13:27
```

## 📁 Project Structure

```
Facial-recognition-attendance/
│
├── images/                # Known faces dataset
├── attendance.csv         # Logged attendance
├── main.py                # Main script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🧩 Future Enhancements

- Add support for cloud-based storage  
- Integrate email/SMS notifications  
- Build a simple web dashboard with Flask or Streamlit  

## 🪪 License

This project is released under the [MIT License](LICENSE).