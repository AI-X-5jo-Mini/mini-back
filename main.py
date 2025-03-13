from face_analysis import extract_face_features
from gpt_api import get_face_analysis, get_compatibility_analysis
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 도메인만 허용하려면 ["https://example.com"] 등 지정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 얼굴 검출을 위한 Cascade Classifier 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_human(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

@app.post("/analyze/")
async def analyze_faces(image1: UploadFile = File(...), image2: UploadFile = File(None)):
    start_time = time.time()
    # 첫 번째 얼굴 특징 추출
    image1_bytes = await image1.read()
    image1_array = np.frombuffer(image1_bytes, np.uint8)
    image1_cv2 = cv2.imdecode(image1_array, cv2.IMREAD_COLOR)
    
    # 첫 번째 이미지 사람/동물 판단
    if not is_human(image1_cv2):
        return JSONResponse(content={"error": " 사람 사진을 업로드해주세요."}, status_code=400)
    
    # 첫 번째 얼굴 특징 추출
    features1 = extract_face_features(image1_cv2)
    if not features1:
        return JSONResponse(content={"error": "첫 번째 얼굴을 감지하지 못했습니다."}, status_code=400)
    
    # 두 번째 사진이 없으면 첫 번째 사람의 단독 분석 수행
    if not image2:
        analysis = get_face_analysis(features1, "이 사람")
        return JSONResponse(content={"analysis": analysis})
    
    # 두 번째 이미지 처리
    image2_bytes = await image2.read()
    image2_array = np.frombuffer(image2_bytes, np.uint8)
    image2_cv2 = cv2.imdecode(image2_array, cv2.IMREAD_COLOR)
    
    # 두 번째 이미지 사람/동물 판단
    if not is_human(image2_cv2):
        return JSONResponse(content={"error": "사람 사진을 업로드해주세요."}, status_code=400)
    
    # 두 번째 얼굴 특징 추출
    features2 = extract_face_features(image2_cv2)
    if not features2:
        return JSONResponse(content={"error": "두 번째 얼굴을 감지하지 못했습니다."}, status_code=400)
    
    # 두 사람의 궁합 분석 수행
    compatibility_result = get_compatibility_analysis(features1, features2, "첫 번째 사람", "두 번째 사람")
    end_time = time.time()
    return JSONResponse(content={"compatibility_result": compatibility_result, "time": end_time - start_time})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

