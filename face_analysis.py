import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_distance

# FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_face_features(image_array):
    """얼굴 특징 추출 (이마, 눈, 코, 입 등)"""
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None  # 얼굴 감지 실패

    face_landmarks = results.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

    # 주요 얼굴 특징 계산
    features = {
        "forehead_width": calculate_distance(landmarks[10], landmarks[151]),  # 이마 넓이
        "forehead_height": calculate_distance(landmarks[10], landmarks[168]),  # 이마 높이
        "eye_size": calculate_distance(landmarks[33], landmarks[263]),  # 눈 크기
        "eye_distance": calculate_distance(landmarks[133], landmarks[362]),  # 눈 사이 거리
        "nose_length": calculate_distance(landmarks[168], landmarks[8]),  # 코 길이
        "nose_width": calculate_distance(landmarks[49], landmarks[279]),  # 콧볼 넓이
        "mouth_width": calculate_distance(landmarks[61], landmarks[291]),  # 입 길이
        "lip_thickness": calculate_distance(landmarks[13], landmarks[14]),  # 입술 두께
        "cheekbone_width": calculate_distance(landmarks[234], landmarks[454]),  # 광대뼈 너비
        "jaw_length": calculate_distance(landmarks[17], landmarks[152]),  # 턱 길이
        "brow_distance": calculate_distance(landmarks[107], landmarks[336])  # 미간 거리
    }

    return features

if __name__ == "__main__":
    features = extract_face_features(cv2.imread("images/face.jpg"))
    print("얼굴 특징:", features)