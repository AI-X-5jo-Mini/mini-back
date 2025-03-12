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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return remove_duplicate_lines(response.choices[0].message.content)


def get_compatibility_analysis(features1, features2, name1="첫 번째 사람", name2="두 번째 사람"):
    """GPT를 사용한 두 사람의 관상 궁합 분석 및 점수 출력"""
    prompt = f"""
    다음 두 사람의 얼굴 특징을 비교하고 관상학적으로 궁합을 분석해줘.

    {name1}의 얼굴 특징:
    - 이마 넓이: {features1['forehead_width']}
    - 이마 높이: {features1['forehead_height']}
    - 눈 크기: {features1['eye_size']}
    - 눈 사이 거리: {features1['eye_distance']}
    - 코 길이: {features1['nose_length']}
    - 콧볼 넓이: {features1['nose_width']}
    - 입 길이: {features1['mouth_width']}
    - 입술 두께: {features1['lip_thickness']}
    - 광대뼈 너비: {features1['cheekbone_width']}
    - 턱 길이: {features1['jaw_length']}
    - 미간 거리: {features1['brow_distance']}

    {name2}의 얼굴 특징:
    - 이마 넓이: {features2['forehead_width']}
    - 이마 높이: {features2['forehead_height']}
    - 눈 크기: {features2['eye_size']}
    - 눈 사이 거리: {features2['eye_distance']}
    - 코 길이: {features2['nose_length']}
    - 콧볼 넓이: {features2['nose_width']}
    - 입 길이: {features2['mouth_width']}
    - 입술 두께: {features2['lip_thickness']}
    - 광대뼈 너비: {features2['cheekbone_width']}
    - 턱 길이: {features2['jaw_length']}
    - 미간 거리: {features2['brow_distance']}


    관상 궁합의 근거
    1. 얼굴형 궁합
    - 비슷한 얼굴형 → 서로 이해도가 높음 (예: 둥근 얼굴 & 둥근 얼굴)
    - 보완적인 얼굴형 → 상호 보완적인 관계 (예: 갸름한 얼굴 & 둥근 얼굴)
    - 반대되는 성향의 얼굴형 → 의견 충돌 가능 (예: 각진 얼굴 & 각진 얼굴)
    
    2. 눈 크기 & 간격
    - 눈 크기가 비슷하면 → 감정 교류가 원활
    - 눈 사이 거리가 적당하면 → 서로에 대한 신뢰가 높음
    - 눈이 너무 가까우면 → 오해가 생기기 쉬움

    3. 코의 크기 & 길이
    - 코 크기가 크면 리더십 강함
    - 코 크기가 비슷하면 협력 잘됨
    - 한쪽이 코가 크고, 한쪽이 작으면 갈등 가능

    4. 입 크기 & 입술 두께
    - 입 크기가 비슷하면 대화가 잘 통함
    - 입술이 두꺼우면 감정 표현이 풍부
    - 입 크기 차이가 크면 대화 방식이 다를 가능성 있음


    위의 정보를 바탕으로 두 사람의 성향, 궁합, 관계의 조화 등을 한국 전통 관상학적 관점에서 분석하고, 궁합 점수를 100점 만점 기준으로 평가해줘.
    한 사람씩 분석 결과와 관상 총 평을 제공해주고, 그 이후에 두 사람의 관상 궁합을 알려주면 좋겠어.
    또한, 점수의 근거는 관상 궁합의 근거를 바탕으로 설명해줘.
    """

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return remove_duplicate_lines(response.choices[0].message.content)
