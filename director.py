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
BGM_DIR = "/home/nyw/AI-practice/resource/pre_train_wav/background" 

# 确保目录存在
for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_VOICE_DIR, VOICE_POOL_DIR, BGM_DIR]:
    os.makedirs(d, exist_ok=True)

# GPU 服务地址
URL_COSY = "http://localhost:8005/generate" # 确保这里端口是你CosyVoice服务的端口
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
        return speech_seg 
    
    try:
        bgm = AudioSegment.from_file(bgm_path)
        
        # 1. 统一基准音量
        bgm = match_target_amplitude(bgm, -20.0)
        
        # 2. 压低背景音 (比人声低 12dB)
        bgm = bgm - 12 
        
        # 3. 循环填充
        target_len = len(speech_seg) + 500
        if len(bgm) < target_len:
            loop_count = (target_len // len(bgm)) + 1
            bgm = bgm * loop_count
            
        # 4. 裁剪
        bgm = bgm[:target_len]
        
        # 5. 淡入淡出
        bgm = bgm.fade_in(500).fade_out(500)
        
        # 6. 叠加
        mixed = speech_seg.overlay(bgm, position=0)
        return mixed

    except Exception as e:
        print(f"⚠️ BGM融合失败 [{os.path.basename(bgm_path)}]: {e}")
        return speech_seg

# ================= 4. LLM 分析逻辑 =================

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
    
    # 修复了这里的字符串拼接和换行问题
    system_prompt = (
        "你是一个有声书脚本制作专家。请将输入的小说文本拆解为 JSON 数组。\n"
        f"可用的背景音乐/音效库如下：{bgm_list_str}\n\n"
        "【核心任务】：\n"
        "将小说原文拆解为适合多人有声剧朗读的脚本。**原文的每一个字、标点都必须保留，不能有任何遗漏！**\n\n"
        "【拆解规则】：\n"
        "1. **对话内容**（引号内）：分配给对应的角色。\n"
        "2. **非对话内容**（引号外）：**全部**分配给角色“旁白”。包括动作、神态、以及“他说”、“道”等引导语。\n"
        "3. **必须拆分**：当一行文字是 [描写 + 对话] 时，必须拆分为 [旁白] + [角色] 两条，不能合并！\n"
        "4. **情绪控制**：情绪 emotion 必须克制（如用'急促'代替'咆哮'，用'低沉'代替'怒吼'）。\n\n"
        "5. 【旁白特殊规则】：旁白是‘说书人’，必须抽离于剧情之外。无论剧情多么激烈（打斗、争吵），旁白的情绪只能是 '沉稳'、'讲述感'、'舒缓' 或 '带有悬念'。严禁给旁白分配 '愤怒'、'哭泣'、'大笑' 等具体的人物情绪！\n\n"
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            temperature=0.01 
        )
        return parse_json_output(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 错误: {e}")
        return []

# ================= 5. 核心流水线 =================

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


def process_pipeline_v2(task_id: str, text: str, user_voice_map: dict):
    TASKS[task_id]["status"] = "analyzing"
    
    print("\n🔍 [1/4] 正在分析文本并分配BGM...")
    dialogues = analyze_novel_roles_llm(text)
    if not dialogues:
        TASKS[task_id]["status"] = "failed"; return

    vm = VoiceManager(VOICE_POOL_DIR)
    
    TASKS[task_id]["status"] = "generating"
    TASKS[task_id]["total"] = len(dialogues)
    
    final_segments = []

    print("\n🗣️ [2/4] 开始生成语音并融合背景音...")
    for i, item in enumerate(dialogues):
        TASKS[task_id]["progress"] = int((i / len(dialogues)) * 100)
        role = item["角色"]
        line = item["台词"]
        raw_emotion = item.get("情绪", "")  
        bgm_filename = item.get("bgm", "")

        # 定义情绪降级映射
        safe_emotion_map = {
            "愤怒": "压抑的怒火，语气冰冷", 
            "咆哮": "咬牙切齿，低沉",
            "大喊": "急促，重音",
            "歇斯底里": "颤抖，哽咽",
            "大笑": "轻笑",
            "狂笑": "得意的笑",
            "悲痛欲绝": "悲伤，低落",
            "恐惧": "紧张，颤音",
            "激昂": "坚定，有力"
        }

        # 处理情绪
        if role == "旁白":
            final_emotion = "沉稳，讲述感，悬疑"
        else:
            final_emotion = raw_emotion
            for danger_key, safe_value in safe_emotion_map.items():
                if danger_key in raw_emotion:
                    print(f"   🛡️ [音色保护] 将 '{raw_emotion}' 降级为 -> '{safe_value}'")
                    final_emotion = safe_value
                    break 

        # 打印日志
        bgm_info = f"🎵 {bgm_filename}" if bgm_filename else "无BGM"
        print(f"\n [{i+1}/{len(dialogues)}] {role}: {line[:15]}... (情绪: {final_emotion}) | {bgm_info}")

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
                # 这里必须传入 final_emotion 让 AI 选角时也知道情绪变了（可选）
                final_wav_path = vm.get_smart_voice(role, final_emotion)
                voice_source_type = "AI自动"

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
                # CosyVoice 通常不需要情绪参数，或者只接受特定参数
                resp = requests.post(URL_COSY, json={"text": line, "speaker": "中文女"}, timeout=60)
            else:
                if final_wav_path and os.path.exists(final_wav_path):
                    # 【重要修改】这里必须使用处理后的 final_emotion，否则音色保护逻辑不生效！
                    resp = requests.post(URL_INDEX, json={
                        "text": line, 
                        "emotion": final_emotion,  # <--- 修改这里：使用 final_emotion
                        "ref_audio_path": final_wav_path
                    }, timeout=60)
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