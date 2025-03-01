import os
import random
from openai import OpenAI
from dotenv import load_dotenv  # Install with: pip install python-dotenv

import jieba  # ✅ Import jieba for Chinese text segmentation
load_dotenv()  # ✅ Load environment variables from .env file
# ✅ 從環境變數讀取 API Key
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)

def generate_adversarial_text(text: str, attack_type: str = "synonym"):
    """用 GPT 生成更強的對抗樣本"""
    
    if attack_type == "synonym":
        # 🔥 讓 GPT 生成語義相似但不同表達的句子
        prompt = f"請改寫以下句子，使其意思相同但用不同的表達方式：\n{text}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    
    
    elif attack_type == "swap":
        words = list(jieba.cut(text))  # ✅ Use jieba to split words correctly
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)  # Pick 2 different word positions
            words[idx1], words[idx2] = words[idx2], words[idx1]  # Swap them

        return "".join(words)  # ✅ Join back into a full sentence
    
    elif attack_type == "contextual":
        # 🔥 GPT 生成一個「更強烈的」語境變化版本
        prompt = (
            f"請將以下句子重新表達，並改變其背景或情境，使其情緒更加強烈或極端，但仍然保持可讀性：\n{text}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,  # 🔥 提高創造力
            max_tokens=100  # ✅ 讓回應更完整
        )
        return response.choices[0].message.content.strip()
    
    else:
        return "⚠️ 無效的攻擊類型"
    


def adversarial_attack_test(text: str, attack_type: str = "synonym"):
    """Generates adversarial examples to test AI robustness."""

    if attack_type == "synonym":
        # 🔥 Generate a semantically similar sentence
        prompt = f"請改寫以下句子，使其意思相同但用不同的表達方式：\n{text}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI API Error: {str(e)}"

    elif attack_type == "swap":
        words = list(jieba.cut(text))  # ✅ Ensure text is split properly
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)  # Pick two positions
            words[idx1], words[idx2] = words[idx2], words[idx1]  # Swap words
            swapped_text = "".join(words)
            
            # Ensure swap actually changes something
            if swapped_text == text:
                return "⚠️ Swap attack failed: No change in text."
            
            return swapped_text
        return "⚠️ Text too short to swap."

    elif attack_type == "contextual":
        # 🔥 Make sentence more extreme or emotionally strong
        prompt = f"請加強以下句子的情緒，使其更加極端或誇張：\n{text}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI API Error: {str(e)}"

    else:
        return "⚠️ Invalid attack type."