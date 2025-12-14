import os
import json
import uuid
import requests
import re
import shutil
import time
import glob
import random
import traceback
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ================= 1. 配置路径 =================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
TEMP_VOICE_DIR = "uploads/custom_voices"
VOICE_POOL_DIR = "/home/nyw/AI-practice/resource/input_audio"
BGM_DIR = "/home/nyw/AI-practice/resource/pre_train_wav/background"  # BGM 库

# 确保目录存在
for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_VOICE_DIR, VOICE_POOL_DIR, BGM_DIR]:
    os.makedirs(d, exist_ok=True)

# GPU 服务地址
URL_COSY = "http://localhost:8005/generate" # 注意你之前的配置端口
URL_INDEX = "http://localhost:8002/generate"

TASKS = {}

# ================= 2. 初始化大模型 =================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-cfc644272f8b4be2aa58f9b240636083"
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ================= 3. 工具函数：音频处理 =================

def match_target_amplitude(sound, target_dBFS=-20.0):
    """将音频响度统一调整到 target_dBFS"""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def mix_speech_with_bgm(speech_seg, bgm_path):
    """
    单句混合逻辑：
    1. 读取 BGM
    2. 循环 BGM 直到覆盖人声长度
    3. 调整 BGM 音量（压低）
    4. 裁剪并做淡入淡出
    5. 混合
    """
    if not bgm_path or not os.path.exists(bgm_path):
        return speech_seg # 没有BGM则原样返回
    
    try:
        bgm = AudioSegment.from_file(bgm_path)
        
        # 1. 统一基准音量
        bgm = match_target_amplitude(bgm, -20.0)
        
        # 2. 压低背景音 (比人声低 12dB，保证人声清晰)
        bgm = bgm - 12 
        
        # 3. 循环填充：如果 BGM 短于人声，进行循环
        # 额外加 500ms 尾韵，防止截断太生硬
        target_len = len(speech_seg) + 500
        if len(bgm) < target_len:
            loop_count = (target_len // len(bgm)) + 1
            bgm = bgm * loop_count
            
        # 4. 精确裁剪
        bgm = bgm[:target_len]
        
        # 5. 淡入淡出 (防止不同BGM切换时的爆音)
        # 开头淡入 500ms，结尾淡出 500ms
        bgm = bgm.fade_in(500).fade_out(500)
        
        # 6. 叠加：BGM 可能会比人声长一点点（尾韵），overlay 会自动扩展长度
        # position=0 表示从头开始叠
        mixed = speech_seg.overlay(bgm, position=0)
        return mixed

    except Exception as e:
        print(f"⚠️ BGM融合失败 [{os.path.basename(bgm_path)}]: {e}")
        return speech_seg

# ================= 4. LLM 分析逻辑 (升级版) =================

def get_all_bgm_filenames():
    """获取 BGM 目录下所有文件名"""
    files = []
    if os.path.exists(BGM_DIR):
        for f in os.listdir(BGM_DIR):
            if f.lower().endswith(('.mp3', '.wav', '.flac')):
                files.append(f)
    return files

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
            bgm = item.get("bgm", "") # 获取 BGM 字段
            
            # 强制统一旁白
            if "旁" in role and "白" in role: role = "旁白"
            if role.lower() == "narrator": role = "旁白"
            
            results.append({"角色": role, "情绪": emotion, "台词": text, "bgm": bgm})
        return results
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")

def analyze_novel_roles_llm(text_content):
    # 1. 获取所有可用的 BGM 文件名
    bgm_files = get_all_bgm_filenames()
    bgm_list_str = json.dumps(bgm_files, ensure_ascii=False)
    
    # 2. 构建 Prompt
    system_prompt = (
        "你是一个有声书导演。请将文本拆解为 JSON 数组。\n"
        f"可用的背景音乐/音效库如下：{bgm_list_str}\n\n"
        "要求：\n"
        "1. 字段包括：role (角色), emotion (情绪), text (台词), bgm (从上述列表中选一个最匹配的文件名，如果没有合适的或不需要，填空字符串)。\n"
        "2. 角色名必须统一。\n"
        "3. 所有旁白的角色名全部统一为“旁白”。\n"
        "4. 严格输出 JSON 格式。\n"
        "5. 【重要】为了保证配音稳定，emotion (情绪) 字段必须保持克制。即使原文描写非常激烈（如歇斯底里、咆哮、大哭），也请转化为相对收敛的描述，例如 '压抑的愤怒'、'冷峻'、'急促'、'低沉'、'哽咽' 等。绝对避免使用会导致声音失真的极端情绪词。"
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
        print(f"❌ LLM 错误: {e}")
        return []

# ================= 5. 核心流水线 (单句融合版) =================

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
# ================= 5. 核心流水线 (带音色日志版) =================

def process_pipeline_v2(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    
    # --- 1. 角色与BGM分析 ---
    print("\n🔍 [1/4] 正在分析文本并分配BGM...")
    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"; return

    vm = VoiceManager(VOICE_POOL_DIR)
    
    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)
    
    final_segments = []

    # --- 2. 逐句生成 + 实时融合 ---
    print("\n🗣️ [2/4] 开始生成语音并融合背景音...")
    for i, item in enumerate(dialogues):
        TASKS[task_id]["progress"] = int((i / len(dialogues)) * 100)
        role = item["角色"]
        line = item["台词"]
        emotion = item.get("情绪", "")
        bgm_filename = item.get("bgm", "")
        
        # 打印当前句子的基本信息
        bgm_info = f"🎵 {bgm_filename}" if bgm_filename else "无BGM"
        print(f"\n➡️ [{i+1}/{len(dialogues)}] {role}: {line[:15]}... | {bgm_info}")

        try:
            # === A. 确定音色逻辑 ===
            final_wav_path = None
            use_cosy_default = False
            voice_source_type = "未知"

            # 1. 尝试用户指定 (精确匹配)
            if role in user_voice_map: 
                final_wav_path = user_voice_map[role]
                voice_source_type = "用户锁定"
            
            # 2. 尝试用户指定 (模糊匹配)
            if not final_wav_path:
                for u_role, u_path in user_voice_map.items():
                    if u_role != "旁白" and role != "旁白" and (u_role in role or role in u_role):
                        final_wav_path = u_path
                        voice_source_type = f"用户模糊({u_role})"
                        break
            
            # 3. 旁白默认逻辑
            if not final_wav_path and role == "旁白": 
                use_cosy_default = True
                voice_source_type = "系统默认"

            # 4. AI 自动选角
            if not final_wav_path and not use_cosy_default: 
                final_wav_path = vm.get_smart_voice(role, emotion)
                voice_source_type = "AI自动"

            # === B. 打印音色选择日志 (这是你想要的功能) ===
            if use_cosy_default:
                print(f"   🎙️ [音色] {voice_source_type} -> CosyVoice (中文女)")
            elif final_wav_path:
                print(f"   🎙️ [音色] {voice_source_type} -> 文件: {os.path.basename(final_wav_path)}")
            else:
                print(f"   ⚠️ [音色] 未找到可用音色，将跳过生成！")
                continue

            # === C. 发送生成请求 ===
            audio_data = None
            
            if use_cosy_default:
                resp = requests.post(URL_COSY, json={"text": line, "speaker": "中文女"}, timeout=60)
            else:
                if final_wav_path and os.path.exists(final_wav_path):
                    resp = requests.post(URL_INDEX, json={"text": line, "emotion": emotion, "ref_audio_path": final_wav_path}, timeout=60)
                else:
                    print(f"   ⚠️ 参考音频文件丢失: {final_wav_path}")
                    continue

            if resp and resp.status_code == 200:
                audio_data = resp.content
            else:
                print(f"   ❌ 生成API报错: Code {resp.status_code if resp else 'None'}")
                continue

            # === D. 音频后处理 (归一化 & BGM) ===
            import io
            speech_seg = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
            speech_seg = match_target_amplitude(speech_seg, -20.0)
            
            # 融合 BGM
            bgm_path = os.path.join(BGM_DIR, bgm_filename) if bgm_filename else None
            if bgm_path and os.path.exists(bgm_path):
                mixed_seg = mix_speech_with_bgm(speech_seg, bgm_path)
            else:
                mixed_seg = speech_seg

            final_segments.append(mixed_seg)
            final_segments.append(AudioSegment.silent(duration=300))

        except Exception as e:
            print(f"   ❌ 处理异常: {e}")
            traceback.print_exc()

    # --- 3. 最终合并 ---
    if not final_segments:
        TASKS[task_id]["status"] = "failed"; return

    print("\n🔨 [3/4] 正在导出最终文件...")
    full_audio = AudioSegment.empty()
    for seg in final_segments:
        full_audio += seg

    final_name = f"{task_id}.mp3"
    full_audio.export(os.path.join(OUTPUT_DIR, final_name), format="mp3")
    
    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["result_url"] = f"/download/{final_name}"
    TASKS[task_id]["progress"] = 100
    print(f"\n🎉 [4/4] 任务完成，文件: {final_name}\n")

# ================= 6. API 接口 =================

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
    print("\n🔍 [DEBUG] 接收前端表单数据:")
    for k, v in form.items():
        if k == "file": continue
        if k.startswith("custom_voice_") and hasattr(v, "filename") and v.filename:
            role = k.replace("custom_voice_", "")
            safe_name = f"{uuid.uuid4()}_{v.filename}"
            save_path = os.path.join(TEMP_VOICE_DIR, safe_name)
            with open(save_path, "wb") as f: shutil.copyfileobj(v.file, f)
            user_voice_map[role] = os.path.abspath(save_path)
            
        elif k.startswith("preset_voice_") and isinstance(v, str) and v:
            role = k.replace("preset_voice_", "")
            path = os.path.join(VOICE_POOL_DIR, v)
            if os.path.exists(path):
                user_voice_map[role] = os.path.abspath(path)

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