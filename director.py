import os
import json
import uuid
import requests
import re
import shutil
import time
import glob
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

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_VOICE_DIR, exist_ok=True)

# GPU 服务地址
URL_COSY = "http://localhost:8001/generate"
URL_INDEX = "http://localhost:8002/generate"

TASKS = {}

# === 2. 初始化大模型 ===
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-cfc644272f8b4be2aa58f9b240636083"
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

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
            
            # 强制统一旁白
            if "旁" in role and "白" in role: role = "旁白"
            if role.lower() == "narrator": role = "旁白"
            
            results.append({"角色": role, "情绪": emotion, "台词": text})
        return results
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return []

def analyze_novel_roles_llm(text_content):
    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个小说拆解专家。请将文本拆解为 JSON 数组：[{'role': 'XX', 'emotion': 'XX', 'text': '...'}]。\n"
                        "规则：1.角色名必须保持统一。2.描写归为'旁白'。3.严格JSON格式。"
                    )
                },
                {"role": "user", "content": text_content}
            ],
            temperature=0.0
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

    def _ask_llm_to_pick(self, role_name, emotion):
        if not self.all_files: return None
        file_map = {os.path.basename(f): f for f in self.all_files}
        prompt = f"角色: {role_name}, 情绪: {emotion}。请从列表 {list(file_map.keys())} 中选一个最合适的文件名，仅输出文件名，没找到输出None。"
        try:
            print(f"🤖 [AI选角] 正在为 {role_name} 挑选...")
            res = client.chat.completions.create(
                model="qwen-max", messages=[{"role": "user", "content": prompt}], temperature=0.1
            )
            picked = res.choices[0].message.content.strip().replace("'", "").replace('"', "")
            if picked in file_map: return file_map[picked]
        except: pass
        return self.all_files[hash(role_name) % len(self.all_files)]

    def get_smart_voice(self, role_name, emotion=""):
        if role_name in self.selection_cache: return self.selection_cache[role_name]
        selected = self._ask_llm_to_pick(role_name, emotion)
        self.selection_cache[role_name] = selected
        return selected

# === 5. 核心流水线 ===
def process_pipeline_v2(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    
    # 🔍 关键调试日志：打印后端收到的“圣旨”
    print("\n" + "="*40)
    print("📋 [DEBUG] 最终生效的用户配置表:")
    if not user_voice_map:
        print("   (空) 用户没有指定任何角色，将全部自动分配")
    for k, v in user_voice_map.items():
        print(f"   🔒 角色[{k}] ===强制绑定===> {os.path.basename(v)}")
    print("="*40 + "\n")

    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"; return

    vm = VoiceManager(VOICE_POOL_DIR)
    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)
    audio_segments = []

    for i, item in enumerate(dialogues):
        TASKS[task_id]["progress"] = int((i / len(dialogues)) * 100)
        role = item["角色"]
        line = item["台词"]
        emotion = item.get("情绪", "")
        
        print(f"➡️ [{i}] {role}: {line[:10]}...")

        try:
            final_wav_path = None
            use_cosy_default = False

            # === 优先级 1: 用户指定 (最强) ===
            # 精确匹配
            if role in user_voice_map:
                final_wav_path = user_voice_map[role]
                print(f"   ✨ [用户] 精确命中: {role} -> {os.path.basename(final_wav_path)}")
            
            # 模糊匹配 (双向包含)
            if not final_wav_path:
                for u_role, u_path in user_voice_map.items():
                    # 排除旁白干扰
                    if u_role != "旁白" and role != "旁白" and (u_role in role or role in u_role):
                        final_wav_path = u_path
                        print(f"   ✨ [用户] 模糊命中: {role} ~= {u_role}")
                        break
            
            # === 优先级 2: 旁白默认 ===
            if not final_wav_path and role == "旁白":
                use_cosy_default = True
                print("   🎙️ [系统] 旁白走默认 CosyVoice")

            # === 优先级 3: AI 自动 ===
            if not final_wav_path and not use_cosy_default:
                final_wav_path = vm.get_smart_voice(role, emotion)
                print(f"   🤖 [AI] 自动分配: {os.path.basename(final_wav_path)}")

            # === 发送请求 ===
            resp = None
            if use_cosy_default:
                resp = requests.post(URL_COSY, json={"text": line, "speaker": "中文女"}, timeout=60)
            else:
                if not final_wav_path or not os.path.exists(final_wav_path):
                    print("   ⚠️ 文件缺失，跳过")
                    continue
                resp = requests.post(URL_INDEX, json={
                    "text": line, "emotion": emotion, "ref_audio_path": final_wav_path
                }, timeout=60)

            if resp and resp.status_code == 200:
                seg_path = os.path.join(OUTPUT_DIR, f"{task_id}_{i}.wav")
                with open(seg_path, "wb") as f: f.write(resp.content)
                audio_segments.append(seg_path)
            else:
                print(f"   ❌ 失败 code={resp.status_code if resp else 'Error'}")

        except Exception as e:
            print(f"   ❌ 异常: {e}")

    # 合并逻辑
    if not audio_segments:
        TASKS[task_id]["status"] = "failed"; return

    combined = AudioSegment.empty()
    for path in audio_segments:
        try:
            combined += AudioSegment.from_wav(path)
            combined += AudioSegment.silent(duration=500)
            os.remove(path)
        except: pass

    final_name = f"{task_id}.mp3"
    combined.export(os.path.join(OUTPUT_DIR, final_name), format="mp3")
    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["result_url"] = f"/download/{final_name}"
    TASKS[task_id]["progress"] = 100
    print("🎉 任务完成")

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
    
    # 1. 提取文本
    file = form.get("file")
    if not file: return JSONResponse(400, {"error": "No file"})
    content = await file.read()
    text = content.decode("utf-8")
    
    # 2. 提取用户配置 (修复版)
    user_voice_map = {}
    
    print("\n🔍 [DEBUG] 接收前端表单数据:")
    for k, v in form.items():
        # 忽略 file 字段，只看 voice 配置
        if k == "file": continue
        
        # === 修复：不再用 isinstance(v, UploadFile) ===
        # 只要对象有 filename 属性，我们就认为它是文件
        if k.startswith("custom_voice_") and hasattr(v, "filename") and v.filename:
            role = k.replace("custom_voice_", "")
            safe_name = f"{uuid.uuid4()}_{v.filename}"
            save_path = os.path.join(TEMP_VOICE_DIR, safe_name)
            
            # 保存上传的文件
            with open(save_path, "wb") as f:
                shutil.copyfileobj(v.file, f)
                
            user_voice_map[role] = os.path.abspath(save_path)
            print(f"   📂 收到文件: [{role}] -> {v.filename}")
            
        elif k.startswith("preset_voice_") and isinstance(v, str) and v:
            role = k.replace("preset_voice_", "")
            path = os.path.join(VOICE_POOL_DIR, v)
            if os.path.exists(path):
                user_voice_map[role] = os.path.abspath(path)
                print(f"   🎵 收到预设: [{role}] -> {v}")

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
    return "<h1>index.html Not Found</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)