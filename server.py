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

# ========== المكت/libs الخارجية ==========
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

# ========== ملف الإعدادات ==========
import config

# وضع التشغيل (web أو cline)
RUN_MODE = os.environ.get("RUN_MODE", "web").lower()
FORCE_CLIENT = os.environ.get("FORCE_CLIENT", "").lower()
MAX_INPUT_TOKENS_CLINE = int(os.environ.get("MAX_INPUT_TOKENS_CLINE", "4096"))

# تحسين تخصيص الذاكرة في CUDA
if config.CUDA_MEMORY_OPTIMIZATION:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# --- إعدادات الهوية والمسارات ---
MODEL_PATH = ""

def find_model_path(base_path):
    """البحث عن مسار النموذج - يدعم المسار المباشر أو snapshot"""
    if not os.path.exists(base_path):
        return None
    
    if os.path.exists(os.path.join(base_path, "config.json")):
        return os.path.abspath(base_path)
    
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir):
        snapshot_folders = [f for f in os.listdir(snapshots_dir) 
                          if os.path.isdir(os.path.join(snapshots_dir, f))]
        if snapshot_folders:
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
    loading_kwargs = {"device_map": {"": "cpu"}, "torch_dtype": torch.float32, "trust_remote_code": config.TRUST_REMOTE_CODE}
elif total_vram_gb > config.VRAM_THRESHOLD_FOR_BFLOAT16:
    loading_kwargs["torch_dtype"] = torch.bfloat16
    print(f"✅ استخدام bfloat16 (VRAM كافية)")
else:
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
    print(f"✅ استخدام 4-bit quantization (توفير VRAM)")

if not config.USE_OPENROUTER:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=config.TRUST_REMOTE_CODE, fix_mistral_regex=config.FIX_MISTRAL_REGEX)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)
    
    if device_count > 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / config.STATIC_FOLDER
INDEX_FILE = STATIC_DIR / "index.html"
DATA_DIR = BASE_DIR / config.DATA_FOLDER
history_filename = config.HISTORY_FILE
if RUN_MODE == "cline":
    history_filename = "history_cline.json"
HISTORY_FILE = DATA_DIR / history_filename

DATA_DIR.mkdir(exist_ok=True)
if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text(json.dumps({}, ensure_ascii=False))

app = FastAPI()
if RUN_MODE != "cline":
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _read_history() -> dict:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {}


def _write_history(payload: dict) -> None:
    if RUN_MODE == "cline":
        return
    HISTORY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

def _get_client_key(request: Request) -> str:
    ip = (request.client.host if request.client else "unknown") or "unknown"
    if FORCE_CLIENT:
        client = FORCE_CLIENT
        return f"{ip}:{client}"
    if RUN_MODE == "cline":
        return f"{ip}:cline"
    ua = (request.headers.get("user-agent") or "").lower()
    if "cline" in ua:
        client = "cline"
    elif "continue" in ua or "vscode" in ua or "copilot" in ua or "cursor" in ua:
        client = "vscode"
    else:
        client = "web"
    return f"{ip}:{client}"


def get_history_for_ip(ip: str):
    data = _read_history()
    return data.get(ip, {})


def get_conversations_for_ip(ip: str) -> list:
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        return []
    convs = user_data.get("conversations", {})
    result = []
    for conv_id, conv_data in convs.items():
        messages = conv_data.get("messages", [])
        title = conv_data.get("title", "محادثة")
        if not title and messages:
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
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return result


def get_conversation(ip: str, conv_id: str) -> dict:
    data = _read_history()
    user_data = data.get(ip, {})
    if isinstance(user_data, list):
        return {}
    return user_data.get("conversations", {}).get(conv_id, {})


def delete_conversation(ip: str, conv_id: str) -> bool:
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
    data = _read_history()
    user_data = data.get(ip, {})
    
    if isinstance(user_data, list):
        user_data = {"conversations": {}}
    
    if "conversations" not in user_data:
        user_data["conversations"] = {}
    
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
            return raw.decode("utf-8-sig")
        except Exception:
            try:
                return raw.decode("utf-16")
            except Exception:
                try:
                    return raw.decode("cp1256")
                except Exception:
                    try:
                        return raw.decode("latin-1")
                    except Exception:
                        return ""
        except Exception:
            return ""

# --- نقاط النهاية (API Endpoints) ---

@app.get("/")
async def get_ui():
    return FileResponse(INDEX_FILE)

@app.get("/v1/history")
async def history(request: Request):
    ip = _get_client_key(request)
    return {"messages": []}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.MODEL_DISPLAY_NAME,
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


@app.get("/v1/conversations")
async def list_conversations(request: Request):
    ip = _get_client_key(request)
    return {"conversations": get_conversations_for_ip(ip)}


@app.get("/v1/conversations/{conv_id}")
async def get_conv(conv_id: str, request: Request):
    ip = _get_client_key(request)
    conv = get_conversation(ip, conv_id)
    return {"messages": conv.get("messages", []), "title": conv.get("title", "")}


@app.delete("/v1/conversations/{conv_id}")
async def delete_conv(conv_id: str, request: Request):
    ip = _get_client_key(request)
    success = delete_conversation(ip, conv_id)
    return {"success": success}


@app.delete("/v1/conversations")
async def delete_all_convs(request: Request):
    ip = _get_client_key(request)
    success = delete_all_conversations(ip)
    return {"success": success}


@app.post("/v1/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        raw = await f.read()
        text = _extract_text(f, raw)
        saved.append({
            "name": f.filename,
            "size": len(raw),
            "content": text[:config.MAX_FILE_CONTENT_LENGTH],
            "content_length": len(text)
        })
    return {"files": saved}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Multi-purpose endpoint - يدعم Streaming و Non-streaming"""
    print(f"📥 Received request to /v1/chat/completions")
    
    try:
        data = await request.json()
        print(f"📝 Messages count: {len(data.get('messages', []))}")
    except Exception as e:
        print(f"❌ Invalid JSON in request body: {e}")
        return JSONResponse(
            {"error": {"message": "Invalid JSON", "type": "invalid_request_error", "param": None, "code": "bad_request"}},
            status_code=400
        )
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    user_agent = (request.headers.get("user-agent") or "").lower()
    accept = (request.headers.get("accept") or "").lower()
    
    # Force disable streaming for Cline and similar tools
    if ("cline" in user_agent or "continue" in user_agent or "vscode" in user_agent or 
        "copilot" in user_agent or "cursor" in user_agent):
        stream = False
        print("🔄 Disabled streaming for Cline/VSCode tool")
    
    if stream and "text/event-stream" not in accept:
        stream = False
    
    conv_id = data.get("conversation_id")
    ip = _get_client_key(request)
    
    # إذا لم يعط conversation_id، استخدم آخر محادثة لهذا ال_IP
    if not conv_id:
        conversations = get_conversations_for_ip(ip)
        if conversations:
            conv_id = conversations[0]["id"]  # استخدم آخر محادثة
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list): 
            content = " ".join([str(i) for i in content])
        cleaned.append({"role": m.get("role"), "content": content})
    
    # Add system prompt
    if cleaned and cleaned[0].get("role") != "system":
        cleaned.insert(0, {
            "role": "system", 
            "content": config.DEFAULT_SYSTEM_PROMPT
        })

    user_text = cleaned[-1].get("content", "") if cleaned else ""
    ip = _get_client_key(request)
    
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
            return JSONResponse(
                {"error": {"message": str(e), "type": "server_error", "param": None, "code": "internal_server_error"}},
                status_code=500
            )
    # ==============================

    # Prepare input
    input_text = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    if RUN_MODE == "cline":
        tokenizer.truncation_side = "left"
        inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS_CLINE
        ).to(model.device)
    else:
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    input_tokens = inputs['input_ids'].shape[1]
    print(f"📊 Input tokens: {input_tokens}")
    
    # Generation parameters
    max_new_tokens = min(config.MAX_NEW_TOKENS, 512)
    if RUN_MODE == "cline":
        max_new_tokens = min(max_new_tokens, 256)
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": config.TEMPERATURE,
        "do_sample": config.DO_SAMPLE,
    }
    if config.TOP_P is not None:
        generation_params["top_p"] = config.TOP_P
    if config.TOP_K is not None:
        generation_params["top_k"] = config.TOP_K
    if config.REPETITION_PENALTY != 1.0:
        generation_params["repetition_penalty"] = config.REPETITION_PENALTY
    
    # Streaming response
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
    
    # Cline-compatible non-streaming response
    else:
        try:
            print(f"🔄 Generating response (non-streaming) for Cline...")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    **generation_params
                )
            
            # Decode the response
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Clean up the response
            full_text = full_text.strip()
            for stop_token in config.STOP_TOKENS:
                if stop_token in full_text:
                    full_text = full_text.split(stop_token)[0]
                    break
            
            # Ensure proper conversation handling
            new_conv_id = conv_id or create_conversation(ip, user_text[:50] if user_text else "محادثة")
            append_exchange(ip, user_text, full_text, new_conv_id)
            
            # Calculate token usage
            prompt_tokens = inputs['input_ids'].shape[1]
            completion_tokens = len(tokenizer.encode(full_text))
            total_tokens = prompt_tokens + completion_tokens
            
            safe_text = full_text if full_text.strip() else " "
            
            # Create Cline-compatible response (chat.completions + responses-like)
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": config.MODEL_DISPLAY_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": safe_text}],
                        "text": safe_text
                    },
                    "text": safe_text,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "system_fingerprint": "fp_1234567890",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": safe_text}]
                    }
                ],
                "output_text": safe_text
            }
            
            print(f"✅ Cline response ready: {len(full_text)} chars, {completion_tokens} tokens")
            return response
        
        except Exception as e:
            print(f"❌ Error during Cline response generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            if config.CLEANUP_MEMORY_AFTER_REQUEST and device_count > 0:
                torch.cuda.empty_cache()
            gc.collect()
            
            return JSONResponse(
                {
                    "error": {
                        "message": f"Internal server error: {str(e)}",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_server_error"
                    }
                },
                status_code=500
            )


if __name__ == "__main__":
    print(f"✅ [ {config.PROJECT_NAME} ] جاهز!")
    print(f"🌐 واجهة الويب متاحة على: http://localhost:{config.SERVER_PORT}")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level=config.LOG_LEVEL)