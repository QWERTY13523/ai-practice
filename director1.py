import os
import json
import uuid
import requests
import re
import shutil
import time
import glob
import math
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === 1. 配置路径 ===
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
TEMP_VOICE_DIR = "uploads/custom_voices"
VOICE_POOL_DIR = "/home/nyw/AI-practice/resource/input_audio"
BGM_POOL_DIR = "/home/nyw/AI-practice/resource/bgm"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_VOICE_DIR, exist_ok=True)
os.makedirs(BGM_POOL_DIR, exist_ok=True)

# GPU 服务地址
URL_COSY = "http://localhost:8001/generate"  # 旁白 (CosyVoice 300M 克隆版)
URL_INDEX = "http://localhost:8002/generate" # 角色 (IndexTTS)

TASKS = {}

# === 2. 初始化大模型 ===
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-cfc644272f8b4be2aa58f9b240636083"
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_available_bgms():
    bgms = []
    if os.path.exists(BGM_POOL_DIR):
        for f in os.listdir(BGM_POOL_DIR):
            if f.lower().endswith(('.wav', '.mp3')):
                bgms.append(f)
    return bgms

# === 3. LLM 分析 ===
def parse_json_output(text_output):
    print(f"----- LLM 原始返回 (前100字) -----\n{text_output[:100]}...\n-------------------------------")
    clean_text = re.sub(r'```json\s*', '', text_output)
    clean_text = re.sub(r'```', '', clean_text).strip()
    try:
        data = json.loads(clean_text)
        results = []
        for item in data:
            role = item.get("role", item.get("角色", "旁白")).strip()
            emotion = item.get("emotion", item.get("情绪", "平淡"))
            text = item.get("text", item.get("台词", ""))
            bgm = item.get("bgm", item.get("背景音", ""))
            
            if "旁" in role and "白" in role: role = "旁白"
            if role.lower() == "narrator": role = "旁白"
            
            results.append({"角色": role, "情绪": emotion, "台词": text, "bgm": bgm})
        return results
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return []

def analyze_novel_roles_llm(text_content):
    bgm_list = get_available_bgms()
    bgm_prompt = ", ".join(bgm_list) if bgm_list else "无"

    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个有声书导演。请拆解文本为 JSON 数组：\n"
                        "[{'role': '角色名', 'emotion': '情绪', 'text': '台词', 'bgm': '背景音文件名'}]\n"
                        "规则：\n"
                        "1. 描写归为'旁白'。\n"
                        "2. 背景音(bgm)：从列表中选择最合适的文件，若不需要填 \"\"。\n"
                        f"   【可用背景音】: [{bgm_prompt}]\n"
                    )
                },
                {"role": "user", "content": text_content}
            ],
            temperature=0.1
        )
        return parse_json_output(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 错误: {e}")
        return []

# === 4. 音色管理器 ===
class VoiceManager:
    def __init__(self, pool_dir):
        self.pool_dir = pool_dir
        self.all_files = []
        if os.path.exists(pool_dir):
            for root, dirs, files in os.walk(pool_dir):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3')):
                        self.all_files.append(os.path.join(root, file))
        self.selection_cache = {}

    def get_smart_voice(self, role_name, emotion=""):
        # 简单缓存策略
        if role_name in self.selection_cache: return self.selection_cache[role_name]
        
        # 简单哈希分配，保证同一角色在不同请求中分配到相同音色
        if self.all_files:
            selected = self.all_files[hash(role_name) % len(self.all_files)]
            self.selection_cache[role_name] = selected
            return selected
        return None

# === 5. 核心流水线 (修复匹配新的 Worker) ===
def process_pipeline_v2(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    
    # 预加载 BGM
    bgm_cache = {}
    if os.path.exists(BGM_POOL_DIR):
        for f in os.listdir(BGM_POOL_DIR):
            if f.lower().endswith(('.wav', '.mp3')):
                try:
                    bgm_cache[f] = AudioSegment.from_file(os.path.join(BGM_POOL_DIR, f)) - 15
                except: pass

    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"; return

    vm = VoiceManager(VOICE_POOL_DIR)
    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)
    
    segments_data = []

    for i, item in enumerate(dialogues):
        TASKS[task_id]["progress"] = int((i / len(dialogues)) * 100)
        role = item["角色"]
        line = item["台词"]
        emotion = item.get("情绪", "")
        bgm_name = item.get("bgm", "")

        print(f"➡️ [{i}] {role}: {line[:10]}... [BGM: {bgm_name}]")

        try:
            # 1. 确定参考音频 (所有角色现在都需要参考音频)
            final_wav_path = None
            
            # 优先从用户配置找
            if role in user_voice_map:
                final_wav_path = user_voice_map[role]
            
            # 其次模糊匹配
            if not final_wav_path:
                for u_role, u_path in user_voice_map.items():
                    if u_role != "旁白" and (u_role in role or role in u_role):
                        final_wav_path = u_path; break
            
            # 最后系统分配 (旁白也必须分配，因为8001变成了克隆接口)
            if not final_wav_path:
                final_wav_path = vm.get_smart_voice(role, emotion)

            # 如果还是没有，且是旁白，强制找一个兜底
            if not final_wav_path and role == "旁白" and vm.all_files:
                final_wav_path = vm.all_files[0]

            if not final_wav_path or not os.path.exists(final_wav_path):
                print(f"   ⚠️ [{role}] 无参考音频，跳过")
                continue

            resp = None
            
            # === 路由与发送 (关键修改) ===
            
            # A. 旁白 -> 8001 (CosyVoice Cross-Lingual)
            # 你的 8001 代码要求: text (Form), prompt_wav (File)
            if role == "旁白":
                with open(final_wav_path, "rb") as audio_file:
                    # 构造 multipart/form-data
                    # 注意：文件名给一个纯英文的 "ref.wav" 防止 Unicode 报错
                    files = {
                        "prompt_wav": ("ref.wav", audio_file, "audio/wav")
                    }
                    data = {
                        "text": line
                    }
                    resp = requests.post(URL_COSY, data=data, files=files, timeout=120)

            # B. 角色 -> 8002 (IndexTTS)
            # 你的 8002 代码要求: JSON {"text":..., "ref_audio_path":...}
            else:
                resp = requests.post(URL_INDEX, json={
                    "text": line, 
                    "emotion": emotion, 
                    "ref_audio_path": final_wav_path # 传路径字符串
                }, timeout=120)

            # === 处理响应 ===
            if resp and resp.status_code == 200:
                seg_path = os.path.join(OUTPUT_DIR, f"{task_id}_{i}.wav")
                with open(seg_path, "wb") as f: f.write(resp.content)
                segments_data.append({"speech_path": seg_path, "bgm_name": bgm_name})
            else:
                print(f"   ❌ API 失败: {resp.status_code if resp else 'None'}")
                if resp: print(f"      Err: {resp.text[:100]}")

        except Exception as e:
            print(f"   ❌ 异常: {e}")

    # === 合成 ===
    if not segments_data:
        TASKS[task_id]["status"] = "failed"; return

    TASKS[task_id]["message"] = "正在混音..."
    combined_audio = AudioSegment.empty()

    for seg in segments_data:
        speech_path = seg["speech_path"]
        bgm_name = seg["bgm_name"]
        
        try:
            speech_audio = AudioSegment.from_wav(speech_path)
            
            if bgm_name and bgm_name in bgm_cache:
                bgm_audio = bgm_cache[bgm_name]
                if len(bgm_audio) > 0:
                    loops = math.ceil(len(speech_audio) / len(bgm_audio))
                    bgm_looped = (bgm_audio * loops)[:len(speech_audio)]
                    mixed = speech_audio.overlay(bgm_looped)
                    combined_audio += mixed
                else:
                    combined_audio += speech_audio
            else:
                combined_audio += speech_audio
            
            combined_audio += AudioSegment.silent(duration=300)
            os.remove(speech_path)

        except Exception as e:
            print(f"⚠️ 混音错误: {e}")
            if os.path.exists(speech_path):
                combined_audio += AudioSegment.from_wav(speech_path)

    final_name = f"{task_id}.mp3"
    combined_audio.export(os.path.join(OUTPUT_DIR, final_name), format="mp3")
    
    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["result_url"] = f"/download/{final_name}"
    TASKS[task_id]["progress"] = 100
    print(f"🎉 任务完成")

# ================= 接口 =================

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    dialogues = analyze_novel_roles_llm(text)
    unique_roles = set(item['角色'] for item in dialogues)
    return {"roles": sorted(list(unique_roles), key=lambda x: 0 if x == "旁白" else 1)}

@app.post("/generate_step")
async def generate_step(request: Request, bg_tasks: BackgroundTasks):
    form = await request.form()
    file = form.get("file")
    if not file: return JSONResponse(400, {"error": "No file"})
    content = await file.read()
    text = content.decode("utf-8")
    
    user_voice_map = {}
    for k, v in form.items():
        if k == "file": continue
        if k.startswith("custom_voice_") and hasattr(v, "filename") and v.filename:
            role = k.replace("custom_voice_", "")
            path = os.path.join(TEMP_VOICE_DIR, f"{uuid.uuid4()}_{v.filename}")
            with open(path, "wb") as f: shutil.copyfileobj(v.file, f)
            user_voice_map[role] = os.path.abspath(path)
        elif k.startswith("preset_voice_") and isinstance(v, str) and v:
            role = k.replace("preset_voice_", "")
            path = os.path.join(VOICE_POOL_DIR, v)
            if os.path.exists(path): user_voice_map[role] = os.path.abspath(path)

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending", "progress": 0}
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
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return f.read()
    return "<h1>Running</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)