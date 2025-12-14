import os
import sys

# ================= 核心配置 (必须放在最开头) =================
# 1. 禁用 DeepSpeed 集成 (防止 Triton/CUDA 报错)
os.environ["HfDeepSpeedConfig"] = "5"

# 2. 强制使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# ========================================================

import torch
import torchaudio
import io
import uvicorn
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

# === 路径配置 ===
# 请确认这个路径是你 CosyVoice 项目的真实路径
COSY_ROOT = "/home/nyw/AI-practice/CosyVoice"
sys.path.append(COSY_ROOT)
sys.path.append(os.path.join(COSY_ROOT, "third_party", "Matcha-TTS"))

# 指定模型路径 (旁白推荐使用 SFT 模型)
MODEL_DIR = os.path.join(COSY_ROOT, "pretrained_models/CosyVoice-300M-SFT")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
except ImportError:
    print("❌ 导入失败: 请检查 COSY_ROOT 路径是否正确。")
    sys.exit(1)

# === 初始化服务 ===
app = FastAPI()

print(f"🚀 [GPU 0] 正在加载 CosyVoice 模型: {MODEL_DIR} ...")

if not os.path.exists(MODEL_DIR):
    print(f"❌ 致命错误: 模型目录不存在: {MODEL_DIR}")
    print("请先从 ModelScope 下载 CosyVoice-300M-SFT 模型。")
    sys.exit(1)

# 加载模型 (SFT模式)
try:
    model = CosyVoice(MODEL_DIR)
    print("✅ CosyVoice 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# === 定义请求格式 (对应 main_server.py 发送的 JSON) ===
class NarratorRequest(BaseModel):
    text: str
    speaker: str = "中文女"  # 默认音色

@app.post("/generate")
def generate(req: NarratorRequest):
    """
    接收 JSON: {"text": "...", "speaker": "中文女"}
    返回: WAV 音频流
    """
    try:
        # SFT 推理 (非流式 stream=False)
        output = model.inference_sft(req.text, req.speaker, stream=False)
        
        # 拼接生成的音频片段
        generated_audio_chunks = [item['tts_speech'] for item in output]
        if not generated_audio_chunks:
            raise HTTPException(status_code=500, detail="生成结果为空")
            
        final_audio = torch.cat(generated_audio_chunks, dim=1)
        
        # 转为 Bytes 返回
        buffer = io.BytesIO()
        # 将 Tensor 转回 CPU 并保存为 WAV
        torchaudio.save(buffer, final_audio.cpu(), model.sample_rate, format="wav")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
        
    except Exception as e:
        print(f"❌ 推理出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "CosyVoice Worker is Running"}

if __name__ == "__main__":
    print("正在启动服务，端口: 8005")
    uvicorn.run(app, host="0.0.0.0", port=8005)