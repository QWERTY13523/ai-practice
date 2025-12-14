import os
import sys
import argparse  # 新增：解析命令行参数
import gradio as gr
import numpy as np

# 解决 macOS 兼容性问题
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 手动添加 ChatTTS 的绝对路径（替换为你的实际路径）
CHATTTS_PATH = "/home/nyw/AI-practice/ChatTTS"  # 务必替换！
sys.path.append(CHATTTS_PATH)

# 从 examples/web 目录导入函数和示例数据
try:
    from examples.web.funcs import (
        load_chat,
        generate_seed,
        on_audio_seed_change,
        refine_text,
        generate_audio,
        logger  # 新增：导入官方日志器
    )
    from examples.web.ex import ex
except ImportError as e:
    print(f"导入失败：{e}")
    sys.exit(1)

# 初始化参数（和官方一致）
voices = {
    "Default": 28532,
    "Female": 8652,
    "Male": 2221,
    "Happy": 10773,
    "Sad": 6098
}
seed_min = 0
seed_max = 100000
use_mp3 = True


def main():
    # 新增：解析命令行参数（完全复制官方逻辑）
    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="server name")
    parser.add_argument("--server_port", type=int, default=7860, help="server port")
    parser.add_argument("--root_path", type=str, help="root path")
    parser.add_argument("--custom_path", type=str, help="custom model path")  # 模型路径参数
    parser.add_argument("--coef", type=str, help="custom dvae coefficient")  # 系数参数
    args = parser.parse_args()

    with gr.Blocks(title="简易 ChatTTS 工具") as demo:
        gr.Markdown("# 简易文本转语音工具")

        text_input = gr.Textbox(
            label="输入文本",
            lines=4,
            value=ex[0][0],
            placeholder="请输入要转换的文本..."
        )

        with gr.Row():
            temperature = gr.Slider(
                minimum=0.00001,
                maximum=1.0,
                value=ex[0][1],
                label="音频随机性"
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=ex[0][2],
                label="top_P"
            )

        generate_btn = gr.Button("生成语音", variant="primary")
        audio_output = gr.Audio(label="生成的语音")
        status_text = gr.Textbox(label="状态", interactive=False)

        generate_btn.click(
            fn=lambda text, temp, tp: generate_audio_flow(text, temp, tp),
            inputs=[text_input, temperature, top_p],
            outputs=[audio_output, status_text]
        )

    # 加载模型（完全复用官方逻辑：通过命令行参数传递 custom_path 和 coef）
    try:
        logger.info("loading ChatTTS model...")  # 使用官方日志
        if load_chat(args.custom_path, args.coef):  # 关键：用命令行参数
            logger.info("Models loaded successfully.")
        else:
            raise Exception("Models load failed.")
    except Exception as e:
        logger.error(f"模型加载错误：{e}")
        sys.exit(1)

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        root_path=args.root_path,
        inbrowser=True
    )


def generate_audio_flow(text, temperature, top_p):
    if not text.strip():
        return None, "请输入文本！"
    try:
        refined_text = refine_text(
            text,
            text_seed=generate_seed(),
            refine_text=True,
            temperature=temperature,
            top_p=top_p,
            split_batch=4
        )
        audio_data = generate_audio(
            refined_text,
            temperature=temperature,
            top_p=top_p,
            spk_emb=on_audio_seed_change(voices["Default"]),
            stream=False,
            audio_seed=voices["Default"],
            sample_text=None,
            sample_audio_code=None,
            split_batch=4
        )
        return audio_data, "生成成功！"
    except Exception as e:
        return None, f"生成失败：{e}"


if __name__ == "__main__":
    main()