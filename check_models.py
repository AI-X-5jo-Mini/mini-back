import openai
import os
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# 사용 가능한 모델 목록 가져오기
models = client.models.list()

print("🔍 사용 가능한 OpenAI 모델 목록:")
for model in models:
    print("-", model.id)
