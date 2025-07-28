import threading
import time
from recorder.capture import start_recording
from agent.audio_agent import build_audio_graph
from agent.video_Agent import build_video_graph
from recorder.capture import analysis_queue
from pathlib import Path

def analyze_audio_file(audio_path):
    graph = build_audio_graph()
    result = graph.invoke({"audio_path": audio_path})

    print("=== 分析完成 ===")
    return {
      "transcript": result.get("transcript"),
      "audio_analysis": result.get("audio_analysis")
    }

a=analyze_audio_file('output/audio_0.wav')
print(a.get("transcript"))
print("-----------------------------------------------------")
print(a.get("audio_analysis"))