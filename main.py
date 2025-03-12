from face_analysis import extract_face_features
from gpt_api import get_face_analysis, get_compatibility_analysis
<<<<<<< HEAD
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
=======

def main(image1, image2=None):
    
    # 첫 번째 얼굴 특징 추출
    features1 = extract_face_features(image1)
    if not features1:
        print("⚠ 첫 번째 얼굴을 감지하지 못했습니다.")
        return
    
    # 두 번째 사진이 없으면 첫 번째 사람의 단독 분석 수행
    if not image2:
        print(f"\n🔮 [{image1}의 관상 분석] 🔮")
        analysis = get_face_analysis(features1, "이 사람")
        print(analysis)
        return
    
    # 두 번째 얼굴 특징 추출
    features2 = extract_face_features(image2)
    if not features2:
        print("⚠ 두 번째 얼굴을 감지하지 못했습니다.")
        return
    
    # 두 사람의 궁합 분석 수행
    print("\n💞 [두 사람의 관상 궁합 분석] 💞")
    compatibility_result = get_compatibility_analysis(features1, features2, "첫 번째 사람", "두 번째 사람")
    print(compatibility_result)

if __name__ == "__main__":
    # 사진이 한 장이면 개인 분석, 두 장이면 궁합 분석 실행
    main("images/jin01.jpg", "images/iu01.jpg")  # 두 장 입력 → 궁합 분석
    # main("images/jin01.jpg")  # 한 장 입력 → 개인 관상 분석
>>>>>>> 8e53b18ea6171eb5215f5d6d3ec59f87b60ccae2
