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
from recorder.capture import start_recording
from agent.audio_agent import build_audio_graph
from agent.video_Agent import build_video_graph
from recorder.buffer import analysis_queue

RECORD_DURATION = 10  # 每段录制 10 秒

def analyze_audio_file():
  pass
def analyze_video_file():
  pass

def analysis_worker():
    while True:
        item = analysis_queue.get()
        if item is None:
            break
        audio_path, video_path = item
        print(f"开始分析: {audio_path}, {video_path}")

        audio_result = analyze_audio_file(audio_path)
        video_result = analyze_video_file(video_path)

        # 可保存分析结果到数据库或文件
        print(f"=== 分析完成 ===\nAudio: {audio_result}\n\nVideo: {video_result}")
        analysis_queue.task_done()

if __name__ == "__main__":
    # 启动分析线程
    t = threading.Thread(target=analysis_worker, daemon=True)
    t.start()

    # 启动录制逻辑
    start_recording(duration=RECORD_DURATION)

    # 等待缓冲区清空
    analysis_queue.join()
