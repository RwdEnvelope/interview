from __future__ import annotations
import base64, pathlib, os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 可按需 import 对应 SDK
from langchain_openai import ChatOpenAI        # OpenAI
import google.generativeai as genai            # Gemini
import dashscope                               # Qwen

# ---------- 固定/可选参数 ----------
PROMPT_ZH = "请观看这段面试视频，用 1-2 句中文简洁描述面试者的动作与表情。"
PROMPT_EN = ("Watch this interview video and describe, in 1–2 concise English sentences, "
             "the candidate’s gestures and facial expressions.")

# ---------- State 类型 ----------
class VState(TypedDict):
    video_path: str
    video_bytes: bytes
    description: str

# ---------- 节点：加载视频 ----------
def node_load(state: VState) -> VState:
    p = pathlib.Path(state["video_path"])
    if not p.exists():
        raise FileNotFoundError(p)
    return {"video_bytes": p.read_bytes()}

# ---------- LLM 调用 ----------
def call_llm(video: bytes, *, provider: str, model: str,
             api_key: str, lang: str) -> str:
    prompt = PROMPT_ZH if lang == "zh" else PROMPT_EN

    if provider == "openai":
        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.2)
        b64 = base64.b64encode(video).decode()
        resp = llm.invoke([{
            "role": "user",
            "content": [
                {"type": "text",  "text": prompt},
                {"type": "video", "video": {"data": b64, "mime_type": "video/mp4"}}
            ]
        }])
        return resp.content.strip()

    if provider == "gemini":
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model)
        up = genai.upload_content(video, mime_type="video/mp4")
        resp = gm.generate_content([prompt, up], generation_config={"temperature": 0.2})
        return resp.text.strip()

    if provider == "qwen":
        out = dashscope.MultiModalConversation.call(
            model=model, api_key=api_key,
            messages=[{"role":"user","content":[
                {"type":"text","content":prompt},
                {"type":"video","content":video}
            ]}]
        )
        return out["output"]["text"].strip()

    raise ValueError("Unsupported provider")

# ---------- 节点：描述 ----------
def make_describe_node(provider: str, model: str,
                       api_key: str, lang: str):
    def _describe(state: VState) -> VState:
        txt = call_llm(state["video_bytes"],
                       provider=provider, model=model,
                       api_key=api_key, lang=lang)
        return {"description": txt}
    return _describe

# ---------- 构建 Graph ----------
def build_graph(*, provider: str, model: str,
                api_key: str, lang: str = "zh"):
    g = StateGraph(VState)
    g.add_node("load", node_load)
    g.add_node("describe",
               make_describe_node(provider, model, api_key, lang))

    g.add_edge(START, "load")
    g.add_edge("load", "describe")
    g.add_edge("describe", END)

    return g.compile()
