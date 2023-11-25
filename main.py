import cv2
import mediapipe as mp
import numpy as np
from fastapi import BackgroundTasks
from fastapi import FastAPI, UploadFile, File
from tensorflow import lite

app = FastAPI()
gloss_list = ['doctor', 'emergency', 'fire', 'firefighter', 'help', 'hurt', 'medicine', 'police']
session_frames_keypoints = {}

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
model_path = 'mediapipe5_pruned.tflite'
interpreter = lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# class Item(BaseModel):
#     features: list


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results


def getKeypoint(frame):
    results = mediapipe_detection(frame, holistic)
    return extract_keypoints(results).astype(np.float32)


def predict_action(frame_sequence):
    input_data = np.expand_dims(frame_sequence, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob_index = np.argmax(res)
    prob = res[max_prob_index]
    return gloss_list[max_prob_index] + str(prob)


def setKeypoint(session_id, frame_data):
    print('backtask')
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    key(session_id, frame)


def key(session_id, frame):
    keypoint = getKeypoint(frame)

    if session_id not in session_frames_keypoints:
        session_frames_keypoints[session_id] = [keypoint]
    else:
        sequences = session_frames_keypoints[session_id]
        if len(sequences) == 15:
            session_frames_keypoints[session_id].pop(0)
        session_frames_keypoints[session_id].append(keypoint)


# @app.post("/process_video/{session_id}")
# async def process_video(session_id: str, frame: UploadFile = File(...)):
#     frame_data = await frame.read()
#
#     if session_id in session_frames_keypoints:
#         keypoints_sequence = session_frames_keypoints[session_id]
#         if len(keypoints_sequence) == 15:
#             action = predict_action(keypoints_sequence)
#             background_tasks.add_task(setKeypoint, session_id, frame_data)
#             return {"predicted_action": action}
#         else:
#             background_tasks.add_task(setKeypoint, session_id, frame_data)
#             return {"predicted_action": "Not enough frames to predict"}


@app.post("/process_video/{session_id}")
async def process_video(session_id: str, frame: UploadFile = File(...),
                        background_tasks: BackgroundTasks = BackgroundTasks()):
    frame_data = await frame.read()
    background_tasks.add_task(setKeypoint, session_id, frame_data)

    if session_id in session_frames_keypoints:
        keypoints_sequence = session_frames_keypoints[session_id]
        print('incoming.......' + str(len(session_frames_keypoints[session_id])))
        if len(keypoints_sequence) == 15:
            action = predict_action(keypoints_sequence)
            return {"predicted_action": action}
    return {"predicted_action": "Not enough frames to predict"}
