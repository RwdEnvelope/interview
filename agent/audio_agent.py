import dashscope
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
load_dotenv()  # 自动读取 .env 文件
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class AudioAnalysisAgent:
    def __init__(self, model: str = "qwen-audio-turbo-latest"):
        self.model = model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = state["audio_path"]
        prompt = (
            "请将这段音频完整转写为文字，"
            "并分析说话者的语速、语调、语气、情绪状态和表达风格。"
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
        except Exception as e:
            return {"error": f"模型响应解析失败: {e}", "raw_response": response}

        transcript = []
        analysis = []

        for item in content:
            if "text" in item:
                text = item["text"].strip()
                if any(k in text for k in ["语速", "语调", "语气", "情绪", "风格"]):
                    analysis.append(text)
                else:
                    transcript.append(text)

        return {
            **state,
            "transcript": "\n".join(transcript),
            "audio_analysis": "\n".join(analysis)
        }


# 定义状态结构
class AudioState(dict):
    """状态字典格式，包含输入和中间输出。"""
    audio_path: str
    transcript: str
    audio_analysis: str

def build_audio_graph():
    builder = StateGraph(AudioState)

    # 加入节点
    builder.add_node("AudioAnalysis", AudioAnalysisAgent())

    # 图的流程
    builder.add_edge(START, "AudioAnalysis")
    builder.add_edge("AudioAnalysis", END)

    return builder.compile()