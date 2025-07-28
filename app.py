import gradio as gr
import threading
import json
from pathlib import Path
from recorder.capture import start_streaming_recording, analysis_queue
from agent.audio_agent import build_audio_graph
from agent.video_Agent import build_video_graph

chat_history = []
chatbot_component = None  # 全局用于刷新

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(text, dict):
            f.write(json.dumps(text, ensure_ascii=False, indent=2))
        else:
            f.write(str(text))

def analyze_audio_file(audio_path):
    graph = build_audio_graph()
    result = graph.invoke({"audio_path": audio_path})
    print("🎤 音频分析完成")
    return {
        "transcript": result.get("transcript"),
        "audio_analysis": result.get("audio_analysis")
    }

def analyze_video_file(video_path):
    graph = build_video_graph()
    result = graph.invoke({"video_path": video_path})
    print("🎥 视频分析完成")
    return {
        "video_analysis": result.get("video_analysis", "（无分析结果）")
    }

def analysis_worker():
    while True:
        item = analysis_queue.get()
        if item is None:
            break
        audio_path, video_path = item
        print(f"📦 分析任务收到: {audio_path}, {video_path}")

        audio_result, video_result = {}, {}

        def run_audio(): nonlocal audio_result; audio_result = analyze_audio_file(audio_path)
        def run_video(): nonlocal video_result; video_result = analyze_video_file(video_path)

        t1 = threading.Thread(target=run_audio)
        t2 = threading.Thread(target=run_video)
        t1.start(); t2.start(); t1.join(); t2.join()

        transcript = audio_result.get("transcript", "（未识别到语音）")
        chat_history.append({"role": "user", "content": transcript})
        chat_history.append({"role": "assistant", "content": f"你刚才说的是：{transcript}"})

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        audio_stem = Path(audio_path).stem
        video_stem = Path(video_path).stem
        save_text(audio_result.get("transcript", ""), out_dir / f"{audio_stem}_transcript.txt")
        save_text(audio_result.get("audio_analysis", ""), out_dir / f"{audio_stem}_analysis.txt")
        save_text(video_result.get("video_analysis", ""), out_dir / f"{video_stem}_analysis.txt")

        print("✅ 分析完成，数据已更新")

        analysis_queue.task_done()

def start_processing():
    threading.Thread(target=analysis_worker, daemon=True).start()
    try:
        start_streaming_recording(chunk_duration=5)
    finally:
        print("🛑 停止采集")
        analysis_queue.put(None)
        analysis_queue.join()

# ========== Gradio UI ==========
def ui():
    global chatbot_component
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="AI 面试官对话", height=600, type="messages")
                chatbot_component = chatbot

                def refresh_chat():
                    return gr.update(value=chat_history)

                refresh_btn = gr.Button("刷新对话")
                refresh_btn.click(fn=refresh_chat, outputs=chatbot)

            with gr.Column(scale=1):
                gr.HTML("""
                    <video id="webcam" width="100%" height="auto" autoplay muted playsinline style="border:1px solid gray"></video>
                    <script>
                    navigator.mediaDevices.getUserMedia({video:true})
                        .then(stream => {
                            document.getElementById('webcam').srcObject = stream;
                        })
                        .catch(err => console.error("摄像头访问失败:", err));
                    </script>
                """)

        demo.load(fn=start_processing)

    return demo

if __name__ == "__main__":
    ui().launch()
