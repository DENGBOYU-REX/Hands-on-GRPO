import requests
from typing import Dict

def get_vllm_advice(url: str, model_name: str, max_tokens: int = 200, temperature: float = 0.2, text: str = None) -> str:
    """
    使用本地部署的 vLLM 模型
    
    :param url: vLLM 服务的 API 地址，例如 "http://localhost:8000/v1/chat/completions"
    :param model_name: 使用的模型名称，例如 "gpt3.5"
    :param text : 输入文本
    :param max_tokens: 最大生成 token 数量
    :param temperature: 温度参数，控制生成的随机性
    :return: 结果
    """

    # 构造请求数据
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are playing helpful assistance."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        # 发送 POST 请求到 vLLM 服务
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 检查请求是否成功

        # 解析返回结果
        result = response.json()
        advice = result["choices"][0]["message"]["content"].strip()
        return advice
    
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with vLLM service: {e}")
        return "check"  # 默认动作
    

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def ref_model(model_path:str, device = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    print('ref model loaded')
    print('ref model device:', model.device)
    return model, tokenizer