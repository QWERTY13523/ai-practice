import os
import re
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
)

def parse_role_lines(text_output):
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
        else:
            print(f" 无法解析行: {line}")
    return results

def analyze_novel_roles(text_content):
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
                    "content": (
                        "请分析以下小说文本，严格按上述规则输出：\n\n"
                        f"{text_content}"
                    )
                }
            ],
            temperature=0.0
        )
        raw_text = completion.choices[0].message.content
        # print(" 模型原始输出：")
        # print(repr(raw_text))
        return parse_role_lines(raw_text)
    except Exception as e:
        print(f" API 调用出错: {e}")
        return []


if __name__ == "__main__":
    file_path = "resource/input_text/孙悟空.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    result = analyze_novel_roles(text)

    # print("\n 解析结果（Python 对象）:")
    # print(result)

    output_dir = "resource/output_text"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "孙悟空.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n JSON 已保存至: {output_path}")