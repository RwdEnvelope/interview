import dashscope
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import re
import json
load_dotenv()  # 自动读取 .env 文件
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def extract_json(text: str) -> str:
    """
    提取模型输出中可能被 markdown 包裹的 JSON 块。
    如果找不到 ```json 包裹，则 fallback 为从第一个 { 开始。
    """
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    # fallback：尝试从第一个 { 位置截断
    return text[text.find("{"):]

class AudioAnalysisAgent:
    def __init__(self, model: str = "qwen-audio-turbo-latest"):
        self.model = model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = state["audio_path"]
        prompt = (
            "你是面试语音分析专家，需要完成两个任务并以结构化 JSON 格式输出：\n"
            "1. transcript: 将音频完整转写为文字，保留自然语言风格。\n"
            "2. audio_analysis: 对音频进行分析，包括以下5个维度：\n"
            "   - 语速：快 / 慢 / 适中，有无波动\n"
            "   - 语调：平稳 / 上扬 / 下沉 / 有强调\n"
            "   - 语气：自信 / 紧张 / 犹豫 / 自然等\n"
            "   - 情绪状态：积极 / 焦虑 / 稳定 / 兴奋等\n"
            "   - 表达风格：条理清晰 / 重复多 / 表达流畅 / 停顿多 等\n\n"
            "请使用以下 JSON 格式严格输出：\n"
            "{\n"
            "  \"transcript\": \"...\",\n"
            "  \"audio_analysis\": {\n"
            "    \"语速\": \"...\",\n"
            "    \"语调\": \"...\",\n"
            "    \"语气\": \"...\",\n"
            "    \"情绪状态\": \"...\",\n"
            "    \"表达风格\": \"...\"\n"
            "  }\n"
            "}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"audio": str(audio_path)},
                    {"text": prompt}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model=self.model,
            messages=messages,
            result_format="message"
        )

        try:
            content = response["output"]["choices"][0]["message"]["content"]
            all_text = "\n".join(item["text"] for item in content if "text" in item)
            print("📋 模型输出：\n", all_text)

            json_text = extract_json(all_text)
            print("📦 解析 JSON：\n", json_text)

            parsed = json.loads(json_text)

            return {
                **state,
                "transcript": parsed.get("transcript", "无回应"),
                "audio_analysis": parsed.get("audio_analysis", "无回应")
            }

        except Exception as e:
            return {
                **state,
                "transcript": "",
                "audio_analysis": {},
                "error": f"模型响应解析失败: {e}",
                "raw_output": response
            }

# 定义状态结构
class AudioState(Dict[str, Any]):
    """状态字典格式，包含输入和中间输出。"""
    audio_path: str
    transcript: str
    audio_analysis: Dict[str, str]

def build_audio_graph():
    builder = StateGraph(AudioState)

    # 加入节点
    builder.add_node("AudioAnalysis", AudioAnalysisAgent())

    # 图的流程
    builder.add_edge(START, "AudioAnalysis")
    builder.add_edge("AudioAnalysis", END)

    return builder.compile()