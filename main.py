from face_analysis import extract_face_features
from gpt_api import get_face_analysis, get_compatibility_analysis
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2

app = FastAPI()

@app.post("/analyze/")
async def analyze_faces(image1: UploadFile = File(...), image2: UploadFile = File(None)):
    # 첫 번째 얼굴 특징 추출
    image1_bytes = await image1.read()
    image1_array = np.frombuffer(image1_bytes, np.uint8)
    image1_cv2 = cv2.imdecode(image1_array, cv2.IMREAD_COLOR)
    features1 = extract_face_features(image1_cv2)
    if not features1:
        return JSONResponse(content={"error": "첫 번째 얼굴을 감지하지 못했습니다."}, status_code=400)
    
    # 두 번째 사진이 없으면 첫 번째 사람의 단독 분석 수행
    if not image2:
        analysis = get_face_analysis(features1, "이 사람")
        return JSONResponse(content={"analysis": analysis})
    
    # 두 번째 얼굴 특징 추출
    image2_bytes = await image2.read()
    image2_array = np.frombuffer(image2_bytes, np.uint8)
    image2_cv2 = cv2.imdecode(image2_array, cv2.IMREAD_COLOR)
    features2 = extract_face_features(image2_cv2)
    if not features2:
        return JSONResponse(content={"error": "두 번째 얼굴을 감지하지 못했습니다."}, status_code=400)
    
    # 두 사람의 궁합 분석 수행
    compatibility_result = get_compatibility_analysis(features1, features2, "첫 번째 사람", "두 번째 사람")
    return JSONResponse(content={"compatibility_result": compatibility_result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

