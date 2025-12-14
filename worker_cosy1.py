import os
import sys
import torch
import torchaudio
import io
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

# === 1. 配置路径 ===
# 请仔细检查你的 CosyVoice 项目路径是否正确
COSY_ROOT = "/home/nyw/AI-practice/CosyVoice"
sys.path.append(COSY_ROOT)
sys.path.append(os.path.join(COSY_ROOT, "third_party", "Matcha-TTS"))

# 【重要】必须指向 300M 基础模型（Clone版），不能是 SFT 版
MODEL_DIR = os.path.join(COSY_ROOT, "pretrained_models/CosyVoice-300M")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("❌ [8001] 导入 CosyVoice 失败，请检查路径配置。")
    sys.exit(1)

app = FastAPI()

# === 2. 加载模型 ===
print(f"🚀 [8001-GPU] 正在加载克隆模型: {MODEL_DIR} ...")
if not os.path.exists(MODEL_DIR):
    print(f"❌ [8001] 致命错误: 找不到模型目录 {MODEL_DIR}")
    print("👉 请下载 CosyVoice-300M 模型（非 SFT）才能使用文件克隆功能。")
    sys.exit(1)

try:
    model = CosyVoice(MODEL_DIR)
    print("✅ [8001] CosyVoice-300M 加载成功！(支持文件上传)")
except Exception as e:
    print(f"❌ [8001] 模型加载失败: {e}")
    sys.exit(1)

# === 3. 定义接口 (Form + File) ===
@app.post("/generate")
async def generate(
    text: str = Form(...),              # 接收文本 (Form表单)
    prompt_wav: UploadFile = File(...)  # 接收文件 (File表单)
):
    """
    接收: text + prompt_wav
    模式: 跨语言/零样本克隆 (无需 prompt_text)
    """
    temp_file = f"temp_narrator_{uuid.uuid4()}.wav"
    try:
        # 1. 保存上传的音频
        content = await prompt_wav.read()
        with open(temp_file, "wb") as f:
            f.write(content)

        # 2. 检查音频有效性
        speech_16k = load_wav(temp_file, 16000)
        if speech_16k.shape[1] < 16000 * 1:
             raise HTTPException(status_code=400, detail="参考音频太短 (<1s)")

        # 3. 推理 (使用 cross_lingual 接口)
        print(f"🎤 [8001]正在克隆旁白: {text[:10]}...")
        output = model.inference_cross_lingual(text, speech_16k, stream=False)

        # 4. 拼接并返回
        chunks = [item['tts_speech'] for item in output]
        if not chunks:
            raise HTTPException(status_code=500, detail="生成结果为空")

        final_audio = torch.cat(chunks, dim=1)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio.cpu(), model.sample_rate, format="wav")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"❌ [8001] 推理出错: {e}")
        # 返回 500 而不是默认的 validation error，防止 unicode 崩溃
        return Response(content=f"Server Error: {str(e)}", status_code=500)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    # 强制运行在 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)