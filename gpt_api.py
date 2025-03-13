import openai
import os
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def remove_duplicate_lines(text):
    lines = text.split("\n")
    seen = set()
    filtered_lines = []
    
    for line in lines:
        if line.strip() not in seen:
            seen.add(line.strip())
            filtered_lines.append(line)

    return "\n".join(filtered_lines)

def get_face_analysis(features, person_name="이 사람"):
    """GPT를 사용한 개별 관상 분석"""
    prompt = f"""
    {person_name}의 얼굴 특징 분석:
    - 이마 넓이: {features['forehead_width']}
    - 이마 높이: {features['forehead_height']}
    - 눈 크기: {features['eye_size']}
    - 눈 사이 거리: {features['eye_distance']}
    - 코 길이: {features['nose_length']}
    - 콧볼 넓이: {features['nose_width']}
    - 입 길이: {features['mouth_width']}
    - 입술 두께: {features['lip_thickness']}
    - 광대뼈 너비: {features['cheekbone_width']}
    - 턱 길이: {features['jaw_length']}
    - 미간 거리: {features['brow_distance']}

    위 정보를 바탕으로 한국 전통 관상학적 해석을 제공해줘.
    """

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return remove_duplicate_lines(response.choices[0].message.content)


def get_compatibility_analysis(features1, features2, name1="첫 번째 사람", name2="두 번째 사람"):
    client = openai.OpenAI(api_key=api_key)

    # 공통되는 얼굴 특징 정보 포맷팅
    face_info1 = f"""
    이마: {features1['forehead_width']}, {features1['forehead_height']}
    눈: {features1['eye_size']}, {features1['eye_distance']}
    코: {features1['nose_length']}, {features1['nose_width']}
    입: {features1['mouth_width']}, {features1['lip_thickness']}
    광대뼈: {features1['cheekbone_width']}
    턱: {features1['jaw_length']}
    미간: {features1['brow_distance']}
    """

    face_info2 = f"""
    이마: {features2['forehead_width']}, {features2['forehead_height']}
    눈: {features2['eye_size']}, {features2['eye_distance']}
    코: {features2['nose_length']}, {features2['nose_width']}
    입: {features2['mouth_width']}, {features2['lip_thickness']}
    광대뼈: {features2['cheekbone_width']}
    턱: {features2['jaw_length']}
    미간: {features2['brow_distance']}
    """

    prompt = f"""
    너는 대한민국 최고의 관상가야.  
    {name1}과 {name2}의 얼굴 특징을 비교하고 전통 관상학을 바탕으로 세 부분으로 나누어 궁합을 분석해줘.  
    결과는 **마크다운이 아닌 일반 텍스트 형식으로 출력해줘.**  
    **응답은 반드시 한국어로 작성해야 해.**

    1️⃣ {name1}의 관상:  
    {face_info1}  
    **6줄 이내로 정리해줘.**  

    2️⃣ {name2}의 관상:  
    {face_info2}  
    **6줄 이내로 정리해줘.**  

    3️⃣ 두 사람의 궁합:  
    - **총점은 반드시 "총점은 ~점입니다." 형식으로 출력해줘.**  
    - 점수는 100점 만점 기준으로 평가해줘.  
    - 종합 점수와 설명을 포함하되, **6줄 이내로 정리해줘.**  
    - **마지막 문장은 반드시 "총점은 ~점입니다." 형식으로 끝나야 합니다.**  

    🔹 **각각의 외모, 성격, 취향, 가치관, 미래 점수를 1~5 범위에서 부여해줘.**  
    🔹 **점수는 반드시 논리적인 기준을 따라야 합니다.**  
    - **5점: 매우 강한 특징 (예: 뚜렷한 개성이 있는 외모, 독보적인 성격)  
    - 4점: 평균 이상 (눈에 띄는 특성이 있음)  
    - 3점: 보통 (크게 튀지 않지만 균형 잡힘)  
    - 2점: 평균 이하 (뚜렷한 장점이 부족한 경우)  
    - 1점: 매우 희미한 특징 (특성이 거의 없는 경우)**  

    🔹 **출력 형식 (고정)**  
    (첫 번째 사람 분석 내용)  
    *** (구분선)  
    (두 번째 사람 분석 내용)  
    *** (구분선)  
    (두 사람의 궁합 분석 내용)  
    마지막 줄은 **"총점은 ~점입니다."** 형식으로 끝내줘.  
    *** (구분선)  
    외모 : {name1} 점수, {name2} 점수
    *** (구분선)  
    성격 : {name1} 점수, {name2} 점수
    *** (구분선)  
    취향 : {name1} 점수, {name2} 점수
    *** (구분선)  
    가치관 : {name1} 점수, {name2} 점수
    *** (구분선)  
    미래 : {name1} 점수, {name2} 점수
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    # 응답 텍스트를 세 부분으로 분리
    full_response = response.choices[0].message.content
    parts = full_response.split('***')

    print("full_response: "+full_response)
    
    
    # 최소 3개의 부분이 있는지 확인하고 안전하게 처리
    if len(parts) >= 3:
        result = {
            "person1_analysis": remove_duplicate_lines(parts[0].strip()),
            "person2_analysis": remove_duplicate_lines(parts[1].strip()),
            "compatibility_analysis": remove_duplicate_lines(parts[2].strip()),
            "score1": remove_duplicate_lines(parts[3].strip()),
            "score2": remove_duplicate_lines(parts[4].strip()),
            "score3": remove_duplicate_lines(parts[5].strip()),
            "score4": remove_duplicate_lines(parts[6].strip()),
            "score5": remove_duplicate_lines(parts[7].strip())
        }
    else:
        # 분리가 제대로 되지 않은 경우 전체 응답을 compatibility_analysis에 저장
        result = {
            "person1_analysis": "",
            "person2_analysis": "",
            "compatibility_analysis": remove_duplicate_lines(full_response)
        }

    return result

