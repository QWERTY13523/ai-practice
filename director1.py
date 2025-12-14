import os
import json
import uuid
import requests
import re
import shutil
import time
import math
import random
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment, effects
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=True)

# === 1. 核心路径配置 ===
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
TEMP_VOICE_DIR = "uploads/custom_voices"

# 资源库路径
VOICE_POOL_DIR = "/home/nyw/AI-practice/resource/pre_train_wav/音频/祥子/情绪"
BGM_POOL_DIR = "/home/nyw/AI-practice/resource/pre_train_wav/background"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_VOICE_DIR, exist_ok=True)

# 服务地址
URL_COSY = "http://localhost:8001/generate"
URL_INDEX = "http://localhost:8002/generate"

TASKS = {}

# === 2. 初始化大模型 ===
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-cfc644272f8b4be2aa58f9b240636083"
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# === 辅助工具 ===
def get_file_list(directory, extensions=('.wav', '.mp3')):
    files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(extensions):
                files.append(f)
    return files


# === 3. LLM 结果解析 ===
def parse_json_output(text_output):
    print(f"\n{'=' * 20} LLM 原始输出 {'=' * 20}\n{text_output}\n{'=' * 50}\n")

    try:
        start_idx = text_output.find('[')
        end_idx = text_output.rfind(']') + 1

        if start_idx == -1 or end_idx == 0:
            print("❌ 无法在输出中找到 JSON 数组结构")
            return []

        clean_text = text_output[start_idx:end_idx]
        data = json.loads(clean_text)

        results = []
        valid_emotions = ["愤怒", "厌恶", "恐惧", "幸福", "悲伤", "惊喜", "激动", "内疚", "自豪", "钦佩", "尴尬",]
        valid_timings = ["start", "middle", "end", "loop"]

        for item in data:
            role = item.get("role", "旁白").strip()
            if role in ["narrator", "Narrator", "", "无"]: role = "旁白"

            emotion = item.get("emotion", "平淡").strip()
            if emotion not in valid_emotions: emotion = "平淡"

            text = item.get("text", "").strip()
            bgm = item.get("bgm", "").strip()
            voice_file = item.get("voice_file", "").strip()

            bgm_timing = item.get("bgm_timing", "start").strip().lower()
            if bgm_timing not in valid_timings: bgm_timing = "start"

            if text:
                results.append({
                    "角色": role,
                    "情绪": emotion,
                    "台词": text,
                    "bgm": bgm,
                    "bgm_timing": bgm_timing,
                    "voice_file": voice_file
                })
        return results
    except Exception as e:
        print(f"❌ JSON 解析异常: {e}")
        return []


# === 4. Prompt 设计 ===
def analyze_novel_roles_llm(text_content):
    bgm_files = get_file_list(BGM_POOL_DIR)
    voice_files = get_file_list(VOICE_POOL_DIR)

    bgm_prompt_list = ", ".join([f"'{f}'" for f in bgm_files]) if bgm_files else "无"
    voice_prompt_list = ", ".join([f"'{f}'" for f in voice_files]) if voice_files else "无"

    system_prompt = (
        "你是一个极其细致的有声书导演。请将小说拆解为 JSON 数组。\n"
        "每个元素包含：{'role': '...', 'emotion': '...', 'text': '...', 'bgm': '...', 'bgm_timing': '...', 'voice_file': '...'}\n\n"
        "【关键规则】：\n"
        "1. **绝不遗漏旁白**：\n"
        "   - 任何未包含在引号（“”）内的文字，必须单独拆分为一条，角色为 '旁白'。\n"
        "2. **背景音(bgm)与时机(bgm_timing)**：\n"
        "   - **bgm**: 从列表中选择最匹配的英文文件名（如 'drop_chopsticks.mp3'），无匹配填 \"\"。\n"
        "     可用列表：[{bgm_prompt_list}]\n"
        "   - **bgm_timing**: ['start', 'middle', 'end', 'loop']。\n"
        "3. **情绪**：[愤怒, 厌恶, 恐惧, 幸福, 悲伤, 惊喜, 激动, 内疚, 自豪, 钦佩, 尴尬, 平淡]\n"
        "4. **音色文件(voice_file)**：\n"
        "   - 从列表中为角色选一个最合适的文件（例如给老爷爷选苍老男声）。\n"
        f"   - 可用列表：[{voice_prompt_list}]\n"
        "5. **输出**：只输出 JSON 数组，无其他废话。\n"
    )

    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            temperature=0.1
        )
        return parse_json_output(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ API 调用错误: {e}")
        return []


# === 5. 处理流水线 ===
def process_pipeline_v2(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    TASKS[task_id]["message"] = "AI 正在细致拆解剧本..."

    # 1. 预加载背景音 (音量增强版)
    bgm_cache = {}
    if os.path.exists(BGM_POOL_DIR):
        for f in os.listdir(BGM_POOL_DIR):
            if f.lower().endswith(('.wav', '.mp3')):
                full_path = os.path.join(BGM_POOL_DIR, f)
                try:
                    # 基础标准化 + 5dB 增益
                    sound = AudioSegment.from_file(full_path)
                    bgm_cache[f] = effects.normalize(sound) + 5
                except:
                    pass

    # 2. 分析
    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["message"] = "分析失败"
        return

    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)

    segments_data = []
    print(f"🚀 开始生成 {len(dialogues)} 个片段...")

    # 获取所有可用音色用于兜底
    all_available_voices = get_file_list(VOICE_POOL_DIR)

    # 3. 生成
    for i, item in enumerate(dialogues):
        progress = int((i / len(dialogues)) * 100)
        TASKS[task_id]["progress"] = progress
        TASKS[task_id]["message"] = f"正在录制: {item['角色']} ({i + 1}/{len(dialogues)})"

        role = item["角色"]
        line = item["台词"]
        emotion = item.get("情绪", "平淡")
        bgm_name = item.get("bgm", "")
        bgm_timing = item.get("bgm_timing", "start")
        llm_voice = item.get("voice_file", "")

        print(f"\n🔵 [句子 {i + 1}] {role}: {line[:15]}...")
        print(f"   └─ 🧠 配置: 情绪[{emotion}] | BGM[{bgm_name} @ {bgm_timing}]")

        try:
            # === 音色选择逻辑 (严格遵循：用户 -> LLM -> 兜底) ===
            ref_wav_path = None
            source_type = "未知"

            # 优先级 1: 用户指定 (最高)
            # user_voice_map 包含了用户上传的文件路径或选择的预设文件路径
            if role in user_voice_map:
                ref_wav_path = user_voice_map[role]
                if "custom_voices" in ref_wav_path:
                    source_type = "⭐ 用户上传"
                else:
                    source_type = "🎹 用户预设"

            # 优先级 2: LLM 匹配 (用户未指定时使用)
            # 必须检查 LLM 推荐的文件是否真的存在于库中
            if not ref_wav_path and llm_voice:
                potential_path = os.path.join(VOICE_POOL_DIR, llm_voice)
                if os.path.exists(potential_path):
                    ref_wav_path = potential_path
                    source_type = "🤖 LLM推荐"

            # 优先级 3: 随机兜底 (前两者都无效时使用)
            if not ref_wav_path and all_available_voices:
                seed = sum(ord(c) for c in role)
                selected = all_available_voices[seed % len(all_available_voices)]
                ref_wav_path = os.path.join(VOICE_POOL_DIR, selected)
                source_type = "🎲 系统兜底"

            # 异常检查
            if not ref_wav_path or not os.path.exists(ref_wav_path):
                print(f"   ❌ [错误] 找不到参考音频 (角色:{role})，跳过此句！")
                continue

            # 日志确认
            print(f"   └─ 💿 [选定] {source_type}: {os.path.basename(ref_wav_path)}")

            # 发送请求
            resp = None
            if role == "旁白":
                with open(ref_wav_path, "rb") as f:
                    files = {"prompt_wav": ("ref.wav", f, "audio/wav")}
                    data = {"text": line}
                    resp = requests.post(URL_COSY, data=data, files=files, timeout=60)
            else:
                payload = {"text": line, "emotion": emotion, "ref_audio_path": ref_wav_path}
                resp = requests.post(URL_INDEX, json=payload, timeout=60)

            if resp and resp.status_code == 200:
                seg_path = os.path.join(OUTPUT_DIR, f"{task_id}_{i}.wav")
                with open(seg_path, "wb") as f:
                    f.write(resp.content)
                segments_data.append({"path": seg_path, "bgm": bgm_name, "timing": bgm_timing})
                print(f"   └─ ✅ 生成成功")
            else:
                print(f"   └─ ❌ 生成失败: {resp.status_code if resp else 'No Response'}")

        except Exception as e:
            print(f"   └─ ❌ 异常: {e}")

    # 4. 合成 (音量优化版)
    if not segments_data:
        TASKS[task_id]["status"] = "failed";
        return

    TASKS[task_id]["message"] = "正在智能混音..."
    full_audio = AudioSegment.empty()

    for seg in segments_data:
        p = seg["path"]
        b = seg["bgm"]
        timing = seg["timing"]

        try:
            voice = AudioSegment.from_wav(p)
            voice = effects.normalize(voice)  # 人声标准化

            if b and b in bgm_cache:
                bgm = bgm_cache[b]  # 已+5dB
                if len(bgm) > 0:
                    # 场景 A: 环境循环 (Loop)
                    if timing == "loop":
                        # 之前是 -12dB，现在改为 -8dB，让环境音更明显一点
                        bgm_loop = bgm - 8
                        loops = math.ceil(len(voice) / len(bgm_loop))
                        bgm_looped = (bgm_loop * loops)[:len(voice)]
                        voice = voice.overlay(bgm_looped)
                        print(f"   🌧️ [Loop] 混入环境: {b}")

                    # 场景 B: 短音效 (SFX)
                    else:
                        # 之前是 -2dB，现在改为 +0dB (原声叠加)，确保响亮
                        bgm_sfx = bgm
                        pos = 0
                        if timing == "start":
                            pos = 0
                        elif timing == "middle":
                            pos = max(0, len(voice) // 2 - len(bgm_sfx) // 2)
                        elif timing == "end":
                            pos = max(0, len(voice) - len(bgm_sfx))

                        voice = voice.overlay(bgm_sfx, position=pos)
                        print(f"   💥 [{timing.upper()}] 插入音效: {b}")

            full_audio += voice
            full_audio += AudioSegment.silent(duration=400)
            os.remove(p)
        except Exception as e:
            if os.path.exists(p): os.remove(p)

    final_name = f"{task_id}.mp3"
    full_audio.export(os.path.join(OUTPUT_DIR, final_name), format="mp3")

    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["result_url"] = f"/download/{final_name}"
    TASKS[task_id]["progress"] = 100
    print(f"\n✅ 全部完成: {final_name}")


# ================= API =================

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    results = analyze_novel_roles_llm(text[:1500])
    roles = set(r['角色'] for r in results)
    return {"roles": sorted(list(roles), key=lambda x: 0 if x == "旁白" else 1)}


@app.post("/generate_step")
async def generate_step(request: Request, bg_tasks: BackgroundTasks):
    form = await request.form()
    file = form.get("file")
    if not file: return JSONResponse(400, {"message": "No file"})
    content = await file.read()
    text = content.decode("utf-8")

    user_voice_map = {}

    # 分类处理表单项
    custom_files = []
    preset_choices = []

    for k, v in form.items():
        if k.startswith("custom_voice_"):
            custom_files.append((k, v))
        elif k.startswith("preset_voice_"):
            preset_choices.append((k, v))

    # 1. 优先处理上传 (最高优先级)
    for k, v in custom_files:
        if hasattr(v, "filename") and v.filename:
            role = k.replace("custom_voice_", "")
            ext = os.path.splitext(v.filename)[1] or ".wav"
            save_name = f"{uuid.uuid4()}{ext}"
            save_path = os.path.join(TEMP_VOICE_DIR, save_name)
            try:
                await v.seek(0)
                with open(save_path, "wb") as f:
                    shutil.copyfileobj(v.file, f)
                user_voice_map[role] = os.path.abspath(save_path)
                print(f"📥 [配置] 角色 [{role}] -> 采用上传文件: {save_path}")
            except Exception as e:
                print(f"❌ [配置] 保存文件失败: {e}")

    # 2. 处理预设 (仅当无上传时生效)
    for k, v in preset_choices:
        if isinstance(v, str) and v:
            role = k.replace("preset_voice_", "")
            if role not in user_voice_map:
                full_path = os.path.join(VOICE_POOL_DIR, v)
                if os.path.exists(full_path):
                    user_voice_map[role] = full_path
                    print(f"👉 [配置] 角色 [{role}] -> 采用预设: {v}")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "analyzing", "progress": 0, "message": "已提交..."}
    bg_tasks.add_task(process_pipeline_v2, task_id, text, user_voice_map)
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def status(task_id: str): return TASKS.get(task_id, {})


@app.get("/download/{name}")
def download(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    return FileResponse(path) if os.path.exists(path) else JSONResponse(404)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index1.html"):
        with open("index1.html", "r", encoding="utf-8") as f: return f.read()
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
    return "<h1>Running</h1>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)