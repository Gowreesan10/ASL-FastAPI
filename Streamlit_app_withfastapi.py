import random
import tempfile

import cv2
import requests
import streamlit as st

FASTAPI_ENDPOINT = "http://localhost:8000"
frame_count = 0

def set_prediction(session_id, frame, output):
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        f"{FASTAPI_ENDPOINT}/process_video/{session_id}", files={"frame": img_encoded.tobytes()}
    )
    result = response.json()

    if result is not None:
        predicted_action = result["predicted_action"]
        if predicted_action is not None:
            output.text(predicted_action)

def main():
    session_id = random.randint(1, 10)
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("ASL Recognition with FastAPI and Streamlit")
    st.caption("Powered by OpenCV, Streamlit")

    option = st.radio("Choose an option:", ("Use Webcam", "Upload Video File"))

    cap = None

    if option == "Use Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            video_path = temp_file.name
            cap = cv2.VideoCapture(video_path)
        else:
            st.warning("Please upload a video file.")

    frame_placeholder = st.empty()
    output = st.empty()
    stop_button_pressed = st.button("Stop")
    global frame_count

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame_count += 1

        if frame_count % 4 == 0:
            set_prediction(session_id, cv2.resize(frame, (224, 224)), output)
        frame_placeholder.image(frame, channels="BGR")
        if stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
