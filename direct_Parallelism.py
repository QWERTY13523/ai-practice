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
import asyncio
import httpx
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
BGM_DIR = "/home/nyw/AI-practice/resource/pre_train_wav/background" 

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_VOICE_DIR, VOICE_POOL_DIR, BGM_DIR]:
    os.makedirs(d, exist_ok=True)

# GPU 服务地址
URL_COSY = "http://localhost:8005/generate" 
URL_INDEX = "http://localhost:8002/generate"

TASKS = {}

# ================= 2. 初始化大模型 =================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-cfc644272f8b4be2aa58f9b240636083"
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ================= 3. 工具函数 =================

def match_target_amplitude(sound, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def mix_speech_with_bgm(speech_seg, bgm_path):
    """
    单句混合逻辑 (修改版)：
    1. 不循环：BGM 只播放一遍。
    2. 如果 BGM 长于人声：裁剪并淡出。
    3. 如果 BGM 短于人声：自然播放结束。
    """
    if not bgm_path or not os.path.exists(bgm_path):
        return speech_seg 
    
    try:
        bgm = AudioSegment.from_file(bgm_path)
        
        # 1. 统一基准音量 & 压低背景音
        bgm = match_target_amplitude(bgm, -20.0)
        bgm = bgm - 12 
        
        # 2. 计算目标长度 (人声 + 500ms 尾韵)
        target_len = len(speech_seg) + 500
        
        # 3. 【核心修改】只播一遍逻辑
        if len(bgm) > target_len:
            # Case A: BGM 比人声长 -> 裁剪到人声长度，并做淡出
            bgm = bgm[:target_len]
            bgm = bgm.fade_out(500)
        else:
            # Case B: BGM 比人声短 -> 不循环，不强行淡出(保留自然尾音)，直接用
            pass
            
        # 统一加开头淡入，防止突兀
        bgm = bgm.fade_in(500)
        
        # 4. 叠加 (如果 BGM 短，overlay 会自动处理，不会报错)
        mixed = speech_seg.overlay(bgm, position=0)
        return mixed

    except Exception as e:
        print(f"⚠️ BGM融合失败 [{os.path.basename(bgm_path)}]: {e}")
        return speech_seg

# ================= 4. LLM 分析逻辑 =================

def get_all_bgm_filenames():
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
            bgm = item.get("bgm", "") 
            if "旁" in role and "白" in role: role = "旁白"
            if role.lower() == "narrator": role = "旁白"
            results.append({"角色": role, "情绪": emotion, "台词": text, "bgm": bgm})
        return results
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return []

def analyze_novel_roles_llm(text_content):
    bgm_files = get_all_bgm_filenames()
    bgm_list_str = json.dumps(bgm_files, ensure_ascii=False)
    system_prompt = (
        "你是一个有声书脚本制作专家。请将输入的小说文本拆解为 JSON 数组。\n"
        f"可用的背景音乐/音效库如下：{bgm_list_str}\n\n"
        "【核心任务】：\n"
        "将小说原文拆解为适合多人有声剧朗读的脚本。**原文的每一个字、标点都必须保留，不能有任何遗漏！**\n\n"
        "【拆解规则】：\n"
        "1. **对话内容**（引号内）：分配给对应的角色。\n"
        "2. **非对话内容**（引号外）：**全部**分配给角色“旁白”。包括动作、神态、以及“他说”、“道”等引导语。\n"
        "3. **必须拆分**：当一行文字是 [描写 + 对话] 时，必须拆分为 [旁白] + [角色] 两条，不能合并！\n"
        "4. **情绪控制**：情绪 emotion 必须克制。尽量不要有愤怒之类比较激动的情绪\n\n"
        "5. 【旁白特殊规则】：旁白是‘说书人’，必须抽离于剧情之外。无论剧情多么激烈，旁白的情绪只能是 '沉稳'、'讲述感'、'舒缓' 或 '带有悬念'。严禁给旁白分配 '愤怒'、'哭泣'、'大笑' 等具体的人物情绪！\n\n"
        "【拆分示例（严格模仿此逻辑）】：\n"
        "输入原文：\n"
        "猪八戒一见，把嘴一噘，嘟囔道：“师父，糟糕了！”\n"
        "输出 JSON：\n"
        "[\n"
        "  {\"role\": \"旁白\", \"emotion\": \"沉稳\", \"text\": \"猪八戒一见，把嘴一噘，嘟囔道：\", \"bgm\": \"\"},\n"
        "  {\"role\": \"猪八戒\", \"emotion\": \"委屈\", \"text\": \"师父，糟糕了！\", \"bgm\": \"funny.mp3\"}\n"
        "]\n\n"
        "输入原文：\n"
        "“快走！”孙悟空一把推开他，“别磨蹭！”\n"
        "输出 JSON：\n"
        "[\n"
        "  {\"role\": \"孙悟空\", \"emotion\": \"急促\", \"text\": \"快走！\", \"bgm\": \"battle.mp3\"},\n"
        "  {\"role\": \"旁白\", \"emotion\": \"讲述感\", \"text\": \"孙悟空一把推开他，\", \"bgm\": \"battle.mp3\"},\n"
        "  {\"role\": \"孙悟空\", \"emotion\": \"急促\", \"text\": \"别磨蹭！\", \"bgm\": \"battle.mp3\"}\n"
        "]\n\n"
        "现在，请处理下面的文本："
    )
    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": text_content}],
            temperature=0.01 
        )
        return parse_json_output(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 错误: {e}")
        return []

# ================= 5. 核心流水线 (异步并发版) =================

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

# --- 单个片段的生成逻辑 (异步) ---
async def generate_segment_async(index, total, item, user_voice_map, vm, semaphore):
    # 使用信号量限制并发数
    async with semaphore:
        role = item["角色"]
        line = item["台词"]
        raw_emotion = item.get("情绪", "")  
        bgm_filename = item.get("bgm", "")

        # --- 情绪安全阀 ---
        safe_emotion_map = {
            "愤怒": "语气冰冷", "咆哮": "咬牙切齿，低沉", "大喊": "急促",
            "歇斯底里": "颤抖，哽咽", "大笑": "轻笑", "狂笑": "得意的笑",
            "悲痛欲绝": "悲伤，低落", "恐惧": "紧张，颤音", "激昂": "坚定，有力"
        }
        
        if role == "旁白":
            final_emotion = "沉稳，讲述感，悬疑"
        else:
            final_emotion = raw_emotion
            for danger_key, safe_value in safe_emotion_map.items():
                if danger_key in raw_emotion:
                    final_emotion = safe_value
                    break 

        print(f"🔄 [{index+1}/{total}] 请求中... {role} ({final_emotion}): {line[:10]}...")

        # --- 选角逻辑 ---
        final_wav_path = None
        use_cosy_default = False
        
        if role in user_voice_map: 
            final_wav_path = user_voice_map[role]
        if not final_wav_path:
            for u_role, u_path in user_voice_map.items():
                if u_role != "旁白" and role != "旁白" and (u_role in role or role in u_role):
                    final_wav_path = u_path; break
        
        if role == "旁白":
            if final_wav_path: use_cosy_default = False 
            else: use_cosy_default = True
        
        if not final_wav_path and not use_cosy_default: 
            final_wav_path = vm.get_smart_voice(role, final_emotion)

        # --- 异步发送 API 请求 ---
        audio_data = None
        async with httpx.AsyncClient(timeout=120.0) as client: 
            try:
                if use_cosy_default:
                    resp = await client.post(URL_COSY, json={"text": line, "speaker": "中文女"})
                else:
                    if final_wav_path and os.path.exists(final_wav_path):
                        resp = await client.post(URL_INDEX, json={
                            "text": line, 
                            "emotion": final_emotion, 
                            "ref_audio_path": final_wav_path
                        })
                    else:
                        print(f"   ❌ 文件丢失: {final_wav_path}")
                        return None

                if resp.status_code == 200:
                    audio_data = resp.content
                    print(f"   ✅ [{index+1}] 生成完毕!")
                else:
                    print(f"   ❌ [{index+1}] API错误: {resp.status_code}")
            except Exception as e:
                print(f"   ❌ [{index+1}] 请求异常: {e}")

        return {
            "index": index,
            "audio_data": audio_data,
            "bgm_filename": bgm_filename 
        }

# --- 主流水线 (异步包装) ---
async def process_pipeline_async(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    print("\n🔍 [1/4] 正在分析文本...")
    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"; return

    vm = VoiceManager(VOICE_POOL_DIR)
    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)

    print(f"\n🚀 [2/4] 启动双卡并发生成! (总句数: {len(dialogues)})")
    
    semaphore = asyncio.Semaphore(4) 
    
    tasks = []
    for i, item in enumerate(dialogues):
        tasks.append(generate_segment_async(i, len(dialogues), item, user_voice_map, vm, semaphore))
    
    results = await asyncio.gather(*tasks)
    
    results = sorted(results, key=lambda x: x["index"] if x else -1)

    print("\n🔨 [3/4] 正在合并音频并添加BGM...")
    final_segments = []
    
    for res in results:
        if not res or not res["audio_data"]:
            continue
            
        try:
            import io
            speech_seg = AudioSegment.from_file(io.BytesIO(res["audio_data"]), format="wav")
            speech_seg = match_target_amplitude(speech_seg, -20.0)
            
            bgm_filename = res["bgm_filename"]
            bgm_path = os.path.join(BGM_DIR, bgm_filename) if bgm_filename else None
            
            if bgm_path and os.path.exists(bgm_path):
                mixed_seg = mix_speech_with_bgm(speech_seg, bgm_path)
            else:
                mixed_seg = speech_seg
                
            final_segments.append(mixed_seg)
            final_segments.append(AudioSegment.silent(duration=300))
            
            TASKS[task_id]["progress"] = int((res["index"] / len(dialogues)) * 100)
            
        except Exception as e:
            print(f"合并出错: {e}")

    if not final_segments:
        TASKS[task_id]["status"] = "failed"; return

    full_audio = AudioSegment.empty()
    for seg in final_segments:
        full_audio += seg

    final_name = f"{task_id}.mp3"
    full_audio.export(os.path.join(OUTPUT_DIR, final_name), format="mp3")
    
    TASKS[task_id]["status"] = "completed"
    TASKS[task_id]["result_url"] = f"/download/{final_name}"
    TASKS[task_id]["progress"] = 100
    print(f"\n🎉 [4/4] 任务完成，文件: {final_name}\n")

def run_async_pipeline(task_id, text, user_voice_map):
    asyncio.run(process_pipeline_async(task_id, text, user_voice_map))

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
            print(f"   📂 收到文件: [{role}] -> {v.filename}")
        elif k.startswith("preset_voice_") and isinstance(v, str) and v:
            role = k.replace("preset_voice_", "")
            path = os.path.join(VOICE_POOL_DIR, v)
            if os.path.exists(path):
                user_voice_map[role] = os.path.abspath(path)
                print(f"   🎵 收到预设: [{role}] -> {v}")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending", "progress": 0}
    bg_tasks.add_task(run_async_pipeline, task_id, text, user_voice_map)
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
    uvicorn.run(app, host="0.0.0.0", port=8039)