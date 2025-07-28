import VideoActionAgent, os

graph = VideoActionAgent.build_graph(
    provider="qwen",
    model="qwen-vl-plus",   # 示例；以你账号可用模型为准
    api_key=os.environ["OPENAI_API_KEY"],
    lang="zh"
)

out = graph.invoke({"video_path": "my_interview_clip.mp4"})
print(out["description"])