from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import analyze_psychology
from adversarial import generate_adversarial_text, adversarial_attack_test
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel
from transformers import pipeline
import openai
import os
import json  # ✅ 確保導入 JSON
import requests 
from dotenv import load_dotenv  # Install with: pip install python-dotenv


# 🔥 關閉 HuggingFace tokenizer 並行處理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()  # ✅ Load environment variables from .env file

# ✅ 從環境變數讀取 API Key
api_key = os.getenv("OPENAI_API_KEY")  

client = OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Sentiment & Psychology NLP API is running!"}


# ✅ **新增對話歷史的存儲結構**
conversation_history = {}  # 存儲每個用戶的對話歷史（key: user_id, value: history）


class TextRequest(BaseModel):
    user_id: str
    text: str
    history: list = []
    deep_dive: bool = False
    attack_type: str = "none"  # ✅ 預設無對抗攻擊
    model: str = "distilbert"  # ✅ 預設 NLP 模型

@app.post("/conversation")
def update_conversation(request: TextRequest):
    """更新對話歷史"""
    user_id = request.user_id
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append(request.text)
    return {"message": "Conversation updated", "history": conversation_history[user_id]}



# ✅ 設定可選擇的 NLP 模型
# ✅ 正確初始化 NLP pipeline，而不是存字串
MODEL_MAPPING = {
    "roberta-large": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-emotion"),  # ✅ 使用 Cardiff NLP 版本
    "bert-base": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"),
    "xlm-roberta": pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment"),
    "gpt-3.5-turbo": "gpt-3.5-turbo"  # ✅ 這個用 OpenAI API 處理，不要用 pipeline()
}


@app.post("/analyze_psychology")
def psychology_analysis(request: dict):
    """使用 GPT-3.5 Turbo 或 HuggingFace 模型進行心理分析"""
    
    model_name = request.get("model", "distilbert")
    user_text = request.get("text", "").strip()
    history = request.get("history", [])

    if not user_text:
        raise HTTPException(status_code=400, detail="請提供有效的文本")

    # ✅ **使用 GPT-3.5 Turbo 進行深度心理分析**
    if model_name == "gpt-3.5-turbo":
        prompt = f"""請分析以下文本的心理狀態，並返回 **JSON 格式**：
        {{
            "state": "主要情緒",
            "confidence": 0-100,
            "emotion_scores": {{
                "Anxiety": 0-100,
                "Confidence": 0-100,
                "Overthinking": 0-100,
                "Self-Doubt": 0-100,
                "Social Avoidance": 0-100,
                "Aggression": 0-100
            }},
            "psychology_factors": {{
                "Anxiety": 0-100,
                "Confidence": 0-100,
                "Overthinking": 0-100,
                "Self-Doubt": 0-100,
                "Social Avoidance": 0-100,
                "Aggression": 0-100
            }},
            "next_question": "請根據這個心理分析結果，提出一個讓使用者深入探索自身情緒的問題"
        }}
        確保所有數值範圍為 0-100，數值越高代表該心理因素越強烈。\n\n
        文本：{user_text}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            analysis_result = json.loads(response.choices[0].message.content)

            return {
                "psychology_analysis": analysis_result,
                "next_question": analysis_result.get("next_question", "這個分析結果對你的影響是什麼？"),
                "model_used": "gpt-3.5-turbo"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GPT-3.5 Turbo Error: {str(e)}")

    # ✅ **使用 Hugging Face 模型**
    if model_name in MODEL_MAPPING:
        try:
            nlp_pipeline = MODEL_MAPPING[model_name]
            analysis = nlp_pipeline(user_text)  # 獲取分類結果
            
            # **確保分析結果是 JSON 格式**
            if isinstance(analysis, list) and isinstance(analysis[0], dict):
                emotion_scores = {entry["label"]: round(entry["score"] * 100, 1) for entry in analysis}

                # ✅ 轉換為 `psychology_factors` 格式
                psychology_factors = {}
                if model_name == "bert-base":
                    psychology_factors = {
                        "Anxiety": emotion_scores.get("1 star", 0) + emotion_scores.get("2 stars", 0),
                        "Confidence": max(emotion_scores.get("4 stars", 0) + emotion_scores.get("5 stars", 0), 50),
                        "Overthinking": emotion_scores.get("1 star", 0),
                        "Self-Doubt": emotion_scores.get("2 stars", 0),
                        "Social Avoidance": emotion_scores.get("1 star", 0),
                        "Aggression": emotion_scores.get("1 star", 0) + emotion_scores.get("2 stars", 0)
                    }
                else:
                    psychology_factors = {
                        "Anxiety": emotion_scores.get("fear", 0) + emotion_scores.get("sadness", 0),
                        "Confidence": max(emotion_scores.get("joy", 0), 50),  # 設定 Confidence 預設值
                        "Overthinking": emotion_scores.get("fear", 0),
                        "Self-Doubt": emotion_scores.get("anger", 0),
                        "Social Avoidance": emotion_scores.get("sadness", 0) + emotion_scores.get("fear", 0),
                        "Aggression": emotion_scores.get("anger", 0)
                    }

                # ✅ 選取主要心理狀態
                primary_state = max(psychology_factors, key=psychology_factors.get)

                # ✅ 🔥 **這裡新增 GPT 來生成 next_question**
                next_q = generate_next_question(primary_state, emotion_scores, history)

                return {
                    "psychology_analysis": {
                        "state": primary_state,
                        "confidence": psychology_factors[primary_state],
                        "emotion_scores": emotion_scores,
                        "psychology_factors": psychology_factors
                    },
                    "next_question": next_q,  # ✅ **這裡的 next_question 是 GPT 生成的**
                    "model_used": model_name
                }
            else:
                raise HTTPException(status_code=500, detail="模型返回的數據格式無法解析")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型執行錯誤: {str(e)}")

    return {"error": "請選擇有效的 NLP 模型"}

@app.post("/generate_question")
def generate_next_question(state: str, history: list, deep_dive: bool = False):
    """
    生成心理學問題：
    - **一般模式**：詢問使用者情緒，挖掘童年陰影
    - **深入模式**：分析「為什麼對方這樣對你？」並提供行動方案
    """
    MAX_ROUNDS = 3
    current_round = len(history)

    if deep_dive:
        # **🔥 深入模式 → 解析行為動機**
        messages = [
            {"role": "system", "content": "你是一位心理學專家，擅長分析家庭關係與代際創傷。"
                                          "請根據使用者的經歷，回答：**為什麼對方會這樣對待他？**\n\n"
                                          "📌 **分析內容應包括：**\n"
                                          "1️⃣ **心理機制（reason）**：對方行為的真正原因。\n"
                                          "2️⃣ **情緒影響（impact）**：這種行為如何影響使用者。\n"
                                          "3️⃣ **行動方案（advice）**：提供 2-3 個實際建議，幫助使用者擺脫這種影響。\n\n"
                                          "⚠️ **請提供 JSON 格式的輸出，不要使用 Markdown。**"},
            {"role": "user", "content": f"當前使用者的情緒是 {state}，他們的回答歷史如下：{history}"},
        ]
        max_token_limit = 300

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=max_token_limit
        )

        # **✅ 嘗試解析 JSON**
        try:
            deep_dive_analysis = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            deep_dive_analysis = {"error": "GPT 返回的 JSON 無法解析", "raw_content": response.choices[0].message.content}

        return deep_dive_analysis  # **✅ 這裡應該回傳完整的 JSON，而不是 `next_question`**

    # **🔥 第 3 輪後 → 生成完整心理學報告**
    elif current_round >= MAX_ROUNDS:
        messages = [
            {"role": "system", "content": "你是一位心理學專家，請根據使用者的對話歷史，"
                                          "生成一份 **完整的心理學分析報告**。\n\n"
                                          "📌 **報告應包括：**\n"
                                          "1️⃣ **核心問題分析**：使用者的情緒根源可能是什麼？\n"
                                          "2️⃣ **心理影響**：這種情緒如何影響他們的行為？\n"
                                          "3️⃣ **解決方案**：提供 2-3 個實際建議來改善情緒。\n\n"
                                          "⚠️ **請確保報告清晰、具體，不超過 300 字，並避免使用專業術語。**"},
            {"role": "user", "content": f"當前使用者的情緒是 {state}，他們的回答歷史如下：{history}"},
        ]
        max_token_limit = 300

    else:
        # **🔥 前 3 輪 → 繼續挖掘童年陰影**
        messages = [
            {"role": "system", "content": "你是一位心理學專家，負責幫助人們深入理解自己的情緒。"
                                          "請根據使用者的情緒與過去的對話，**提出一個能讓他們回憶童年陰影的問題**。"
                                          "請確保問題 **直接、簡短、不超過 50 字**，並且 **讓使用者聯想到童年經歷**。"
                                          "**只輸出問題，不提供額外說明**。"},
            {"role": "user", "content": f"當前使用者的情緒是 {state}，他們的回答歷史如下：{history}"},
        ]
        max_token_limit = 50

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=max_token_limit
    )

    return response.choices[0].message.content.strip()

class AttackRequest(BaseModel):
    text: str
    attack_type: str = "synonym"
    model: str = "distilbert"  # ✅ 新增 NLP 模型選擇參數

@app.post("/adversarial_attack")
def adversarial_test(request: AttackRequest):
    try:
        adversarial_text = generate_adversarial_text(request.text, request.attack_type)

        original_analysis = simple_psychology_analysis({"text": request.text, "model": request.model})
        adversarial_analysis = (
            simple_psychology_analysis({"text": adversarial_text, "model": request.model})
            if adversarial_text != request.text
            else original_analysis
        )

        return {
            "original_text": request.text,
            "adversarial_text": adversarial_text,
            "original_analysis": original_analysis,
            "adversarial_analysis": adversarial_analysis
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing adversarial attack: {str(e)}")
    

# 在 simple_psychology_analysis API 內部
@app.post("/simple_psychology_analysis")
def simple_psychology_analysis(request: dict):
    model_name = request.get("model", "distilbert")
    user_text = request.get("text", "").strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="請提供有效的文本")

    if model_name == "gpt-3.5-turbo":
        prompt = f"""請分析以下文本的心理狀態，並返回 JSON 格式：
        {{
            "state": "主要情緒",
            "confidence": 0-100,
            "emotion_scores": {{
                "Anxiety": 0-100,
                "Confidence": 0-100,
                "Overthinking": 0-100,
                "Self-Doubt": 0-100,
                "Social Avoidance": 0-100,
                "Aggression": 0-100
            }},
            "psychology_factors": {{
                "Anxiety": 0-100,
                "Confidence": 0-100,
                "Overthinking": 0-100,
                "Self-Doubt": 0-100,
                "Social Avoidance": 0-100,
                "Aggression": 0-100
            }}
        }}
        文本：{user_text}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            analysis_result = json.loads(response.choices[0].message.content)

            return {
                "state": analysis_result["state"],
                "confidence": analysis_result["confidence"],
                "emotion_scores": analysis_result["emotion_scores"],
                "psychology_factors": analysis_result["psychology_factors"],
                "model_used": "gpt-3.5-turbo"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GPT-3.5 Turbo Error: {str(e)}")

    elif model_name in MODEL_MAPPING:
        try:
            nlp_pipeline = MODEL_MAPPING[model_name]
            analysis = nlp_pipeline(user_text)
            
            emotion_scores = {entry["label"]: round(entry["score"] * 100, 1) for entry in analysis}

            psychology_factors = {}
            
            if model_name == "xlm-roberta":
                psychology_factors = {
                    "Anxiety": emotion_scores.get("negative", 0),
                    "Confidence": max(emotion_scores.get("positive", 0), 50),
                    "Overthinking": emotion_scores.get("negative", 0),
                    "Self-Doubt": emotion_scores.get("negative", 0),
                    "Social Avoidance": emotion_scores.get("negative", 0),
                    "Aggression": emotion_scores.get("negative", 0),
                }
            elif model_name == "bert-base":
                psychology_factors = {
                    "Anxiety": emotion_scores.get("1 star", 0) + emotion_scores.get("2 stars", 0),
                    "Confidence": max(emotion_scores.get("4 stars", 0) + emotion_scores.get("5 stars", 0), 50),
                    "Overthinking": emotion_scores.get("1 star", 0),
                    "Self-Doubt": emotion_scores.get("2 stars", 0),
                    "Social Avoidance": emotion_scores.get("1 star", 0),
                    "Aggression": emotion_scores.get("1 star", 0) + emotion_scores.get("2 stars", 0)
                }
            else:
                psychology_factors = {
                    "Anxiety": emotion_scores.get("fear", 0) + emotion_scores.get("sadness", 0),
                    "Confidence": max(emotion_scores.get("joy", 0), 50),
                    "Overthinking": emotion_scores.get("fear", 0),
                    "Self-Doubt": emotion_scores.get("anger", 0),
                    "Social Avoidance": emotion_scores.get("sadness", 0) + emotion_scores.get("fear", 0),
                    "Aggression": emotion_scores.get("anger", 0)
                }

            primary_state = max(psychology_factors, key=psychology_factors.get)

            return {
                "state": primary_state,
                "confidence": psychology_factors[primary_state],
                "emotion_scores": emotion_scores,
                "psychology_factors": psychology_factors,
                "model_used": model_name
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型執行錯誤: {str(e)}")

    raise HTTPException(status_code=400, detail="請選擇有效的 NLP 模型")