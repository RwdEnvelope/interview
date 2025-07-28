# from dotenv import load_dotenv
# import os
# import dashscope
# from agent.audio_agent import build_audio_graph
# from agent.video_Agent import build_video_graph
# #------------------------------------------------------------------
# # # ✅ 加载 .env 文件中的 API Key
# # load_dotenv()
# # dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# # # ✅ 构造音频路径
# # audio_path = "agent/assets/audio.mp3"
# # if not os.path.exists(audio_path):
# #     raise FileNotFoundError(f"找不到音频文件: {audio_path}")

# # # ✅ 将本地路径转成 file:// URI
# # from pathlib import Path
# # audio_uri = str(Path(audio_path).resolve())
# # print(audio_uri)

# # # ✅ 调用分析 Agent
# # agent = AudioAnalysisAgent()
# # result = agent({"audio_path": audio_uri})

# # # ✅ 输出结果
# # print("=== 音频分析完成 ===")
# # print(f"- 对话内容：{result.get('transcript')}")
# # print(f"- 语音特征总结：\n{result.get('audio_analysis')}")
# #---------------------------------------------------------------

# if __name__ == "__main__":
#     graph = build_audio_graph()
#     result = graph.invoke({"audio_path": "agent/assets/audio.mp3"})

#     print("=== 分析完成 ===")
#     print(result.get("transcript", "（无分析结果）"))
#     print('\n\n\n\n\n')
#     print(result.get("audio_analysis", "（无分析结果）"))

#     graph = build_video_graph()
#     result = graph.invoke({"video_path": "agent/assets/video.mp4"})

#     print("=== 分析完成 ===")
#     print(result.get("video_analysis", "（无分析结果）"))
import threading
import time
from recorder.capture import start_streaming_recording, analysis_queue
from agent.audio_agent import build_audio_graph
from agent.video_Agent import build_video_graph
from pathlib import Path
import json


def analyze_audio_file(audio_path):
    graph = build_audio_graph()
    result = graph.invoke({"audio_path": audio_path})
    print("=== 音频分析完成 ===")
    return {
        "transcript": result.get("transcript"),
        "audio_analysis": result.get("audio_analysis")
    }

def analyze_video_file(video_path):
    graph = build_video_graph()
    result = graph.invoke({"video_path": video_path})
    print("=== 视频分析完成 ===")
    return {
        "video_analysis": result.get("video_analysis", "（无分析结果）")
    }

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(text, dict):
            f.write(json.dumps(text, ensure_ascii=False, indent=2))
        else:
            f.write(str(text))


def analysis_worker():
    while True:
        item = analysis_queue.get()
        if item is None:
            break
        audio_path, video_path = item
        print(f"开始分析: {audio_path}, {video_path}")

        audio_result = {}
        video_result = {}

        def analyze_audio():
            nonlocal audio_result
            audio_result = analyze_audio_file(audio_path)

        def analyze_video():
            nonlocal video_result
            video_result = analyze_video_file(video_path)

        t1 = threading.Thread(target=analyze_audio)
        t2 = threading.Thread(target=analyze_video)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        audio_stem = Path(audio_path).stem
        video_stem = Path(video_path).stem

        save_text(audio_result.get("transcript", ""), out_dir / f"{audio_stem}_transcript.txt")
        save_text(audio_result.get("audio_analysis", ""), out_dir / f"{audio_stem}_analysis.txt")
        save_text(video_result.get("video_analysis", ""), out_dir / f"{video_stem}_analysis.txt")

        print(f"✅ 分析结果已保存：\n"
              f" - {audio_stem}_transcript.txt\n"
              f" - {audio_stem}_analysis.txt\n"
              f" - {video_stem}_analysis.txt\n")

        analysis_queue.task_done()

if __name__ == "__main__":
    # 启动分析线程
    threading.Thread(target=analysis_worker, daemon=True).start()

    # 启动流式采集
    try:
        start_streaming_recording(chunk_duration=5)  # 每段 5 秒
    finally:
        print("🛑 停止采集，等待分析线程处理完毕")
        analysis_queue.put(None)
        analysis_queue.join()