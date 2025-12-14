# server_text.py
import os
import re
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# 初始化 DashScope 客户端
# 请确保环境变量 DASHSCOPE_API_KEY 已设置
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

app = FastAPI()


class AnalyzeRequest(BaseModel):
    text: str


def parse_role_lines(text_output):
    """ 解析大模型返回的文本，提取角色、情绪和台词 """
    text_output = text_output.replace('：', ':').replace('；', ';')
    lines = text_output.strip().split('\n')
    results = []

    pattern = r'^\[角色:([^;]+);情绪:([^;]+);台词:(.*?)\]$'

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            role, emotion, dialogue = match.groups()
            results.append({
                "角色": role.strip(),
                "情绪": emotion.strip(),
                "台词": dialogue.strip()
            })
    return results


@app.post("/analyze")
def analyze_novel(req: AnalyzeRequest):
    """
    接收小说文本，调用 Qwen 模型进行角色台词拆分
    """
    print(f"收到分析请求，文本长度: {len(req.text)}")

    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的小说文本分析助手，任务是将小说片段严格划分为两类：\n"
                        "1. 【角色台词】：仅包括用引号（“”）括起来的直接对话，或明确以‘XX说：’引导的话语。\n"
                        "2. 【旁白】：包括所有叙述、描写、心理活动（即使写‘他想：...’，只要没用引号，就算旁白）。\n\n"
                        "输出规则：\n"
                        "- 每行一个条目，格式：[角色:XX;情绪:XX;台词:XXXX]\n"
                        "- 角色只能是具体人名或‘旁白’\n"
                        "- 情绪必须从以下11个词中选择：愤怒、厌恶、恐惧、幸福、悲伤、惊喜、激动、内疚、自豪、钦佩、尴尬\n"
                        "- 台词必须**完全忠实原文**，不要添加引号、不要改写、不要合并句子\n"
                        "- 如果一段旁白包含多种情绪，选择最主要的一种\n"
                        "- **绝对不要编造对话！不要给旁白加引号！**"
                    )
                },
                {
                    "role": "user",
                    "content": f"请分析以下小说文本，严格按上述规则输出：\n\n{req.text}"
                }
            ],
            temperature=0.0
        )

        raw_text = completion.choices[0].message.content
        parsed_data = parse_role_lines(raw_text)

        return {"result": parsed_data}

    except Exception as e:
        print(f"API 调用错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 使用 argparse 支持 --port 参数
    parser = argparse.ArgumentParser(description="Text Analysis API Service")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口 (默认: 8000)")
    args = parser.parse_args()

    print(f"正在启动文本分析服务，端口: {args.port}")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)