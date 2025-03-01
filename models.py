from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv  # Install with: pip install python-dotenv

load_dotenv()  # ✅ Load environment variables from .env file

api_key = os.getenv("OPENAI_API_KEY")  

client = OpenAI(api_key=api_key)
# 初始化 VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str):
    """使用 VADER & TextBlob 進行情感分析"""
    vader_score = analyzer.polarity_scores(text)["compound"]
    blob_score = TextBlob(text).sentiment.polarity
    sentiment = "Positive" if vader_score > 0.05 else "Negative" if vader_score < -0.05 else "Neutral"
    return {"sentiment": sentiment, "vader_score": vader_score, "blob_score": blob_score}

# 載入 Hugging Face 預訓練的心理分析 NLP 模型
psychology_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
# 使用 Hugging Face 預訓練的 NLP pipeline 來分析情緒
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", return_all_scores=True)

def analyze_psychology(text: str):
    """心理分析 NLP 模型"""
    # 取得情緒分數
    emotion_scores = emotion_analyzer(text)[0]

    # 將情緒分數轉換成 JSON 格式
    emotions = {entry['label']: entry['score'] for entry in emotion_scores}

    # ✅【完整心理分析指標，確保所有指標都有值】✅
    psychology_factors = {
        "Anxiety": round((emotions.get("fear", 0) + emotions.get("sadness", 0.1)) * 100, 1),
        "Confidence": round(max(emotions.get("joy", 0), 0.5) * 100, 1),
        "Overthinking": round(emotions.get("fear", 0) * 100, 1),
        "Self-Doubt": round(emotions.get("anger", 0) * 100, 1),
        "Social Avoidance": round((emotions.get("sadness", 0.1) + emotions.get("fear", 0.2)) * 100, 1),
        "Aggression": round(emotions.get("anger", 0) * 100, 1)
    }

    # ✅【確保所有心理指標即使是 0 也會返回！】✅
    for key in psychology_factors:
        if key not in psychology_factors:
            psychology_factors[key] = 0

    # 選擇最高的心理狀態
    primary_psychology_state = max(psychology_factors, key=psychology_factors.get)

    return {
        "state": primary_psychology_state,
        "confidence": psychology_factors[primary_psychology_state],
        "emotion_scores": emotions,
        "psychology_factors": psychology_factors
    }


def generate_next_question(state: str, history: list, deep_dive: bool = False):
    """生成心理分析问题，支持「深入真相模式」，揭露隐藏心理动机"""

    MAX_ROUNDS = 3  # 設定最大對話輪數
    current_round = len(history)

    if deep_dive:
        # **🔥「深入真相模式」- 解析對方行為背後的心理動機**
        messages = [
            {"role": "system", "content": "你是一位心理學專家，擅長分析人際關係與深層心理動機。"
                                          "請根據使用者的經歷，回答：**為什麼對方會這樣對待他？**"
                                          "不要提供溫和的安慰，而是直擊核心，揭露對方的心理機制與行為動機。"
                                          "請確保內容直接、銳利，並且提供可以改變使用者價值觀的關鍵建議。"},
            {"role": "user", "content": f"當前使用者的情緒是 {state}，他們的回答歷史如下：{history}"},
        ]

        max_token_limit = 250  # 🔥 允许更长的解析
    elif current_round >= MAX_ROUNDS:
        # **第 N 輪後，生成完整心理學報告**
        messages = [
            {"role": "system", "content": "你是一位心理學專家，負責幫助人們理解自己的情緒。"
                                          "請根據使用者的對話歷史，生成一份 **完整的心理學分析報告**。"
                                          "報告應該包括：\n"
                                          "1️⃣ **核心問題分析**：使用者的情緒根源可能是什麼？\n"
                                          "2️⃣ **心理影響**：這種情緒如何影響他們的行為、決策？\n"
                                          "3️⃣ **解決方案**：提供 2-3 個實際可行的建議，幫助使用者改善情緒。\n"
                                          "請確保報告內容清晰、具體，**不超過 250 字**，不要使用過於專業的術語，讓一般人能理解。"},
            {"role": "user", "content": f"當前使用者的情緒是 {state}，他們的回答歷史如下：{history}"},
        ]

        max_token_limit = 250
    else:
        # **前 N 輪 → 繼續問深入的問題**
        messages = [
            {"role": "system", "content": "你是一位心理學專家，負責幫助人們深入理解自己的情緒。"
                                          "請根據使用者的情緒與過去的對話，**提出一個能讓他們回憶童年陰影的問題**。"
                                          "請確保問題 **直接、簡短、不超過 50 字**，"
                                          "並且 **讓使用者聯想到童年經歷**。**只輸出問題，不提供額外說明**。"},
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
