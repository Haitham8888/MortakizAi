# ========== المكتبات المدمجة في Python ==========
import os
import requests
import io
import json
import re
import uuid
import zipfile
import gc
from datetime import datetime
from pathlib import Path
from threading import Thread

# ========== المكتبات الخارجية ==========
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

# ========== ملف الإعدادات ==========
import config

# تحسين تخصيص الذاكرة في CUDA
if config.CUDA_MEMORY_OPTIMIZATION:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- إعدادات الهوية والمسارات ---
MODEL_PATH = ""

def find_model_path(base_path):
    """البحث عن مسار النموذج - يدعم المسار المباشر أو snapshot"""
    if not os.path.exists(base_path):
        return None
    
    # تحقق إذا كان المسار يحتوي على config.json مباشرة
    if os.path.exists(os.path.join(base_path, "config.json")):
        return os.path.abspath(base_path)
    
    # ابحث عن snapshots
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir):
        # احصل على أحدث snapshot
        snapshot_folders = [f for f in os.listdir(snapshots_dir) 
                          if os.path.isdir(os.path.join(snapshots_dir, f))]
        if snapshot_folders:
            # استخدم أول snapshot (أو يمكنك ترتيبهم حسب التاريخ)
            snapshot_path = os.path.join(snapshots_dir, snapshot_folders[0])
            if os.path.exists(os.path.join(snapshot_path, "config.json")):
                print(f"📦 تم العثور على snapshot: {snapshot_folders[0]}")
                return os.path.abspath(snapshot_path)
    
    return None

for p in config.MODEL_PATH_OPTIONS:
    found_path = find_model_path(p)
    if found_path:
        MODEL_PATH = found_path
        break

if not MODEL_PATH and not config.USE_OPENROUTER:
    raise RuntimeError("❌ لم يتم العثور على مجلد الموديل.")

# 1. فحص العتاد
device_count = torch.cuda.device_count()

# تطبيق حد عدد الكروت من الإعدادات
if config.MAX_GPU_COUNT is not None and device_count > config.MAX_GPU_COUNT:
    device_count = config.MAX_GPU_COUNT
    
total_vram_gb = 0

if device_count > 0:
    print(f"🚀 [ {config.PROJECT_NAME} ] يبدأ العمل...")
    print(f"🖥️ عدد كروت الشاشة المستخدمة: {device_count}")
    
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        total_vram_gb += vram_gb
        print(f"   GPU {i}: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    
    print(f"📊 إجمالي VRAM: {total_vram_gb:.1f} GB")
else:
    print(f"🚀 [ {config.PROJECT_NAME} ] يبدأ العمل...")
    print(f"⚠️ لا يوجد GPU متاح - سيتم استخدام CPU")

# 2. استراتيجية التحميل
loading_kwargs = {"device_map": "auto", "trust_remote_code": config.TRUST_REMOTE_CODE}
if device_count == 0:
    # CPU فقط
    loading_kwargs = {"device_map": {"": "cpu"}, "torch_dtype": torch.float32, "trust_remote_code": config.TRUST_REMOTE_CODE}
elif total_vram_gb > config.VRAM_THRESHOLD_FOR_BFLOAT16:
    # VRAM كافية جداً - استخدام bfloat16 للدقة العالية
    loading_kwargs["torch_dtype"] = torch.bfloat16
    print(f"✅ استخدام bfloat16 (VRAM كافية)")
else:
    # VRAM محدودة - استخدام 4-bit quantization (توفير أقصى)
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
    print(f"✅ استخدام 4-bit quantization (توفير VRAM)")

if not config.USE_OPENROUTER:
    # 3. تحميل الموديل والـ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=config.TRUST_REMOTE_CODE, fix_mistral_regex=config.FIX_MISTRAL_REGEX)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)
    
    # تنظيف الذاكرة بعد التحميل
    if device_count > 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / config.STATIC_FOLDER
INDEX_FILE = STATIC_DIR / "index.html"
DATA_DIR = BASE_DIR / config.DATA_FOLDER
HISTORY_FILE = DATA_DIR / config.HISTORY_FILE

DATA_DIR.mkdir(exist_ok=True)
if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text(json.dumps({}, ensure_ascii=False))

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _read_history() -> dict:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {}


def _write_history(payload: dict) -> None:
    HISTORY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

def _get_ip(request: Request) -> str:
    return (request.client.host if request.client else "unknown") or "unknown"


def get_history_for_ip(ip: str):
    data = _read_history()
    return data.get(ip, {})


def get_conversations_for_ip(ip: str) -> list:
    """Get list of conversations for an IP"""
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        # Migrate old format to new
        return []
    convs = user_data.get("conversations", {})
    result = []
    for conv_id, conv_data in convs.items():
        messages = conv_data.get("messages", [])
        title = conv_data.get("title", "محادثة")
        if not title and messages:
            # Use first user message as title
            for msg in messages:
                if msg.get("role") == "user":
                    title = msg.get("content", "")[:50]
                    break
        result.append({
            "id": conv_id,
            "title": title or "محادثة",
            "created_at": conv_data.get("created_at"),
            "message_count": len(messages)
        })
    # Sort by created_at descending
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return result


def get_conversation(ip: str, conv_id: str) -> dict:
    """Get a specific conversation"""
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        return {}
    return user_data.get("conversations", {}).get(conv_id, {})


def delete_conversation(ip: str, conv_id: str) -> bool:
    """Delete a conversation"""
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        return False
    convs = user_data.get("conversations", {})
    if conv_id in convs:
        del convs[conv_id]
        data[ip] = user_data
        _write_history(data)
        return True
    return False


def delete_all_conversations(ip: str) -> bool:
    """Delete all conversations for an IP"""
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        data[ip] = {"conversations": {}}
    else:
        user_data["conversations"] = {}
        data[ip] = user_data
    _write_history(data)
    return True


def create_conversation(ip: str, title: str = "محادثة") -> str:
    """Create a new empty conversation and return its ID"""
    data = _read_history()
    user_data = data.get(ip, {})
    
    if isinstance(user_data, list):
        user_data = {"conversations": {}}
    
    if "conversations" not in user_data:
        user_data["conversations"] = {}
    
    conv_id = str(uuid.uuid4())[:8]
    user_data["conversations"][conv_id] = {
        "created_at": datetime.now().isoformat(),
        "title": title[:50] if title else "محادثة",
        "messages": []
    }
    
    data[ip] = user_data
    _write_history(data)
    return conv_id


def append_exchange(ip: str, user_text: str, assistant_text: str, conv_id: str = None) -> str:
    """Append messages to a conversation, returns conversation ID"""
    data = _read_history()
    user_data = data.get(ip, {})
    
    # Migrate old format
    if isinstance(user_data, list):
        user_data = {"conversations": {}}
    
    if "conversations" not in user_data:
        user_data["conversations"] = {}
    
    # Create new conversation if needed
    if not conv_id:
        conv_id = str(uuid.uuid4())[:8]
    
    conv = user_data["conversations"].get(conv_id)
    if not conv:
        conv = {
            "created_at": datetime.now().isoformat(),
            "title": user_text[:50] if user_text else "محادثة",
            "messages": []
        }
        user_data["conversations"][conv_id] = conv
    
    if user_text:
        conv["messages"].append({"role": "user", "content": user_text})
    if assistant_text:
        conv["messages"].append({"role": "assistant", "content": assistant_text})
    
    data[ip] = user_data
    _write_history(data)
    return conv_id


def _extract_docx_text(raw: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open("word/document.xml") as f:
                xml = f.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", "", xml)
        return text
    except Exception:
        return ""


def _extract_text(file: UploadFile, raw: bytes) -> str:
    name = file.filename or "file"
    ctype = file.content_type or ""
    if name.lower().endswith(".docx") or "wordprocessingml" in ctype:
        return _extract_docx_text(raw)
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            return raw.decode("latin-1")
        except Exception:
            return ""

# --- نقاط النهاية (API Endpoints) ---

@app.get("/")
async def get_ui():
    return FileResponse(INDEX_FILE)

@app.get("/v1/history")
async def history(request: Request):
    ip = _get_ip(request)
    return {"messages": []}


@app.get("/v1/conversations")
async def list_conversations(request: Request):
    ip = _get_ip(request)
    return {"conversations": get_conversations_for_ip(ip)}


@app.get("/v1/conversations/{conv_id}")
async def get_conv(conv_id: str, request: Request):
    ip = _get_ip(request)
    conv = get_conversation(ip, conv_id)
    return {"messages": conv.get("messages", []), "title": conv.get("title", "")}


@app.delete("/v1/conversations/{conv_id}")
async def delete_conv(conv_id: str, request: Request):
    ip = _get_ip(request)
    success = delete_conversation(ip, conv_id)
    return {"success": success}


@app.delete("/v1/conversations")
async def delete_all_convs(request: Request):
    ip = _get_ip(request)
    success = delete_all_conversations(ip)
    return {"success": success}


@app.post("/v1/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        raw = await f.read()
        text = _extract_text(f, raw)
        # لا يتم حفظ الملفات على القرص - فقط قراءة في الذاكرة
        saved.append({
            "name": f.filename,
            "size": len(raw),
            "content": text[:config.MAX_FILE_CONTENT_LENGTH]
        })
    return {"files": saved}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Multi-purpose endpoint - يدعم Streaming و Non-streaming"""
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)  # تحقق إذا كان streaming مطلوب
    conv_id = data.get("conversation_id")
    
    # تنظيف الرسائل
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list): 
            content = " ".join([str(i) for i in content])
        cleaned.append({"role": m.get("role"), "content": content})
    
    # إضافة system prompt للحفاظ على formatting الكود
    if cleaned and cleaned[0].get("role") != "system":
        cleaned.insert(0, {
            "role": "system", 
            "content": config.DEFAULT_SYSTEM_PROMPT
        })

    user_text = cleaned[-1].get("content", "") if cleaned else ""
    ip = _get_ip(request)
    
    # === OPENROUTER INTEGRATION ===
    if config.USE_OPENROUTER:
        try:
            print(f"🔄 Using OpenRouter: {config.OPENROUTER_MODEL_ID}")
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000"
            }
            payload = {
                "model": config.OPENROUTER_MODEL_ID,
                "messages": cleaned,
                "stream": stream,
                "temperature": config.TEMPERATURE
            }
            
            if stream:
                def iter_openrouter():
                    nonlocal conv_id
                    full_resp = ""
                    # Ensure conversation exists
                    if not conv_id:
                        conv_id = create_conversation(ip, user_text[:50] if user_text else "محادثة")
                        yield f"data: {json.dumps({'conversation_id': conv_id})}\n\n"
                    
                    try:
                        with requests.post(config.OPENROUTER_API_URL, json=payload, headers=headers, stream=True) as r:
                            if r.status_code == 401:
                                error_msg = "🚨 **خطأ في المصادقة (401):** مفتاح OpenRouter API غير صحيح أو منتهي الصلاحية. يرجى التحقق من المتغير `OPENROUTER_API_KEY` في ملف `server.py`."
                                print(f"❌ {error_msg}")
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': error_msg}}]})}\n\n"
                                return

                            r.raise_for_status()
                            for line in r.iter_lines():
                                if line:
                                    txt = line.decode('utf-8')
                                    if txt.startswith("data: ") and "[DONE]" not in txt:
                                        try:
                                            chunk = json.loads(txt[6:])
                                            if "choices" in chunk and chunk["choices"]:
                                                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                                                full_resp += delta
                                        except:
                                            pass
                                    yield txt + "\n\n"
                    except Exception as e:
                        error_text = f"\n\n❌ **OpenRouter Error:** {str(e)}"
                        print(error_text)
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': error_text}}]})}\n\n"
                    
                    append_exchange(ip, user_text, full_resp, conv_id)
                    yield "data: [DONE]\n\n"

                return StreamingResponse(iter_openrouter(), media_type="text/event-stream")
            
            else:
                resp = requests.post(config.OPENROUTER_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data_resp = resp.json()
                
                ai_text = data_resp["choices"][0]["message"]["content"]
                conv_id = conv_id or create_conversation(ip, user_text[:50] if user_text else "محادثة")
                append_exchange(ip, user_text, ai_text, conv_id)
                
                return data_resp

        except Exception as e:
            print(f"❌ OpenRouter Error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    # ==============================

    # تحضير الإدخال
    input_text = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    # إعداد معاملات التوليد
    generation_params = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "do_sample": config.DO_SAMPLE,
    }
    if config.TOP_P is not None:
        generation_params["top_p"] = config.TOP_P
    if config.TOP_K is not None:
        generation_params["top_k"] = config.TOP_K
    if config.REPETITION_PENALTY != 1.0:
        generation_params["repetition_penalty"] = config.REPETITION_PENALTY
    
    # إذا كان streaming، استخدم streaming response
    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, streamer=streamer, **generation_params)
        
        def generate_with_cleanup():
            try:
                model.generate(**generation_kwargs)
            finally:
                if config.CLEANUP_MEMORY_AFTER_REQUEST and device_count > 0:
                    torch.cuda.empty_cache()
                gc.collect()
        
        Thread(target=generate_with_cleanup).start()
        full_text = ""
        new_conv_id = conv_id

        async def generate():
            nonlocal full_text, new_conv_id
            if not conv_id:
                new_conv_id = create_conversation(ip, user_text[:50] if user_text else "محادثة")
                yield f"data: {json.dumps({'conversation_id': new_conv_id})}\n\n"
            
            for text in streamer:
                if await request.is_disconnected():
                    break
                if not text.strip():
                    continue
                full_text += text
                if any(s in text for s in config.STOP_TOKENS):
                    break
                yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
            
            append_exchange(ip, user_text, full_text, new_conv_id or conv_id)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # Non-streaming response (للـ Cline)
    else:
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **generation_params
                )
            
            # فك التشفير مع الحفاظ على التنسيق والمسافات
            full_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False  # الحفاظ على المسافات الأصلية
            )
            
            # توقف عند علامات التوقف
            for stop_token in config.STOP_TOKENS:
                if stop_token in full_text:
                    full_text = full_text.split(stop_token)[0]
                    break
            
            # حفظ المحادثة
            new_conv_id = conv_id or create_conversation(ip, user_text[:50] if user_text else "محادثة")
            append_exchange(ip, user_text, full_text, new_conv_id)
            
            # تنظيف الذاكرة
            if config.CLEANUP_MEMORY_AFTER_REQUEST and device_count > 0:
                torch.cuda.empty_cache()
            gc.collect()
            
            # إرجاع الرد بصيغة OpenAI القياسية
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": config.MODEL_DISPLAY_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs['input_ids'].shape[1],
                    "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
                    "total_tokens": outputs.shape[1]
                }
            }
        
        except Exception as e:
            if config.CLEANUP_MEMORY_AFTER_REQUEST and device_count > 0:
                torch.cuda.empty_cache()
            gc.collect()
            return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print(f"✅ [ {config.PROJECT_NAME} ] جاهز!")
    print(f"🌐 واجهة الويب متاحة على: http://localhost:{config.SERVER_PORT}")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level=config.LOG_LEVEL)