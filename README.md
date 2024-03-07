## ASL Recognition with Streamlit and FastAPI

This repository contains code for a real-time American Sign Language (ASL) recognition application built using Streamlit for the frontend and FastAPI for the backend. The application utilizes MediaPipe for pose estimation and a pre-trained TensorFlow Lite model for ASL classification.

### Files

1. **main.py**: This file contains the Streamlit application code. It allows users to use their webcam or upload a video file for ASL recognition. The application sends video frames to the FastAPI backend for processing and displays the predicted ASL action.

2. **app.py**: This file contains the FastAPI backend code. It defines routes for processing video frames uploaded by the Streamlit application. The backend performs pose estimation using MediaPipe and predicts ASL actions using a TensorFlow Lite model.

3. **mediapipe50.tflite**: This file contains the pre-trained TensorFlow Lite model for ASL recognition. It is used by the FastAPI backend to predict ASL actions based on the detected keypoints.

### Dependencies

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- FastAPI
- TensorFlow Lite
- Streamlit

### Setup Instructions

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI backend:

```bash
uvicorn app:app --reload
```

4. Run the Streamlit application:

```bash
streamlit run main.py
```

5. Access the application in your web browser using the provided URL.

### Usage

- Upon running the Streamlit application, users can choose between using their webcam or uploading a video file for ASL recognition.
- The application sends video frames to the FastAPI backend, which performs pose estimation and predicts ASL actions based on the detected keypoints.
- Predicted ASL actions are displayed in real-time on the Streamlit interface.

### Credits

- This project utilizes MediaPipe for pose estimation and TensorFlow Lite for ASL classification.
- Streamlit and FastAPI are used for building the web application and backend API, respectively.
