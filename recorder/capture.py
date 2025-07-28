import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
from pathlib import Path
import keyboard

exit_flag = threading.Event()
analysis_queue = queue.Queue()

AUDIO_SR = 16000
CHUNK_DURATION = 5
AUDIO_FRAME_COUNT = AUDIO_SR * CHUNK_DURATION
VIDEO_CHUNK_FRAMES = 20 * CHUNK_DURATION  # 20FPS * 5s = 100帧

def wait_for_exit():
    keyboard.wait('q')
    print("🛑 收到退出信号")
    exit_flag.set()

def audio_stream_worker(audio_buffer, frame_count, save_dir):
    idx = 0
    current_audio = []
    accumulated_frames = 0

    def callback(indata, frames, time_info, status):
        nonlocal current_audio, idx, accumulated_frames
        current_audio.append(indata.copy())
        accumulated_frames += frames

        if accumulated_frames >= frame_count:
            audio_chunk = np.concatenate(current_audio, axis=0)
            audio_path = f"{save_dir}/audio_{idx}.wav"
            sf.write(audio_path, audio_chunk, AUDIO_SR)
            audio_buffer.put(audio_path)
            current_audio.clear()
            accumulated_frames = 0
            idx += 1

    with sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=callback):
        while not exit_flag.is_set():
            time.sleep(0.1)


def video_stream_worker(video_buffer, frame_target, save_dir):
    cap = cv2.VideoCapture(0)
    idx = 0
    frame_buffer = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    while cap.isOpened() and not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        cv2.imshow('🎥 Recording Stream', frame)

        if len(frame_buffer) >= frame_target:
            video_path = f"{save_dir}/video_{idx}.mp4"
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for f in frame_buffer:
                out.write(f)
            out.release()
            video_buffer.put(video_path)
            frame_buffer.clear()
            idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag.set()
            break

    cap.release()
    cv2.destroyAllWindows()

def start_streaming_recording(chunk_duration=CHUNK_DURATION, save_dir="output"):
    Path(save_dir).mkdir(exist_ok=True)
    audio_buffer = queue.Queue()
    video_buffer = queue.Queue()

    threading.Thread(target=wait_for_exit, daemon=True).start()
    threading.Thread(target=audio_stream_worker, args=(audio_buffer, chunk_duration * AUDIO_SR, save_dir), daemon=True).start()
    threading.Thread(target=video_stream_worker, args=(video_buffer, chunk_duration * 20, save_dir), daemon=True).start()

    while not exit_flag.is_set():
        try:
            audio_path = audio_buffer.get(timeout=1)
            video_path = video_buffer.get(timeout=1)
            print(f"📦 提交分析: {audio_path}, {video_path}")
            analysis_queue.put((audio_path, video_path))
        except queue.Empty:
            continue

    print("🛑 录制结束，等待分析完成")
    analysis_queue.put(None)
