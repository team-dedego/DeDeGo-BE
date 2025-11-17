from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DEDEGO(판교어 번역기) API", version="1.0.0")

# CORS 설정
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://dedego.yuuka.me",
    "https://dedego.vercel.app",
    "https://dedego.kro.kr"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

class TranslateRequest(BaseModel):
    text: str
    direction: Literal["to_pangyo", "to_korean"]

class TermExplanation(BaseModel):
    term: str
    meaning: str
    original: str = ""

class TranslateResponse(BaseModel):
    original: str
    translated: str
    direction: str
    terms: List[TermExplanation]

PROMPT_TEMPLATES = {
    "to_pangyo": """
당신은 판교 IT 업계에서 사용하는 "판교어" 전문가입니다.

다음 일반 한국어 문장을 자연스러운 판교어로 번역하고, 사용된 판교어 용어들을 설명해주세요.

**입력 문장:**
{text}

**지침:**
1. 자연스럽고 실제 판교에서 사용할 법한 표현으로 번역
2. 영어 비즈니스 용어를 적절히 섞어서 사용하되 영어를 사용하면 안됨 (예: ASAP → 아삽)
3. 한국어 조사/어미는 유지하되 핵심 명사/동사는 영어로 대체
4. 과하게 어렵지 않게, 실무에서 실제 쓰일 법한 수준으로

**응답 형식 (반드시 JSON으로만 응답):**
{{
  "translated": "번역된 판교어 문장",
  "terms": [
    {{
      "term": "사용된 판교어 용어",
      "meaning": "해당 용어의 의미 1줄 정도로 간단히 설명",
      "original": "원어 (예: ASAP, Follow-up 등)"
    }}
  ]
}}

JSON 외 다른 텍스트는 절대 포함하지 마세요.
""",
    
    "to_korean": """
당신은 판교 IT 업계에서 사용하는 "판교어" 전문가입니다.

다음 판교어 문장을 일반인도 이해할 수 있는 표준 한국어로 번역하고, 문장에 포함된 판교어 용어들을 설명해주세요.

**입력 문장:**
{text}

**지침:**
1. 모든 판교어 용어를 표준 한국어로 자연스럽게 번역.
2. 일반 직장인이 쉽게 이해할 수 있는 표현 사용
3. 비즈니스 맥락은 유지하되 쉬운 언어로 풀어서 설명

**응답 형식 (반드시 JSON으로만 응답):**
{{
  "translated": "번역된 표준 한국어 문장",
  "terms": [
    {{
      "term": "원문에 있던 판교어 용어",
      "meaning": "해당 용어의 의미 1줄 정도로 간단히 설명",
      "original": "원어 (예: ASAP, Follow-up 등)"
    }}
  ]
}}

JSON 외 다른 텍스트는 절대 포함하지 마세요.
"""
}

@app.get("/api/")
async def root():
    """API 상태 체크"""
    return {
        "service": "DEDEGO(판교어 번역기) API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "translate": "POST /api/translate",
            "health": "GET /api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}

@app.post("/api/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    DEDEGO(판교어 번역) API
    
    - to_pangyo: 일반 한국어 → 판교어
    - to_korean: 판교어 → 일반 한국어
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="텍스트를 입력해주세요")
        
        prompt = PROMPT_TEMPLATES[request.direction].format(text=request.text)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Markdown code block 제거
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json\n", "").replace("```\n", "").replace("```", "")
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패. 응답: {response_text}")
            raise HTTPException(
                status_code=500, 
                detail=f"LLM 응답을 파싱할 수 없습니다. 다시 시도해주세요."
            )
        
        return TranslateResponse(
            original=request.text,
            translated=result.get("translated", ""),
            direction=request.direction,
            terms=[
                TermExplanation(**term) 
                for term in result.get("terms", [])
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"번역 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)