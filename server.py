import torch
import uvicorn
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import io
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# --- إعدادات الهوية والمسارات ---
PROJECT_NAME = "مَرْتَكَز - MortakizAi"
path_options = ["./models_cache/models--Qwen--Qwen2.5-Coder-7B-Instruct", "./qwen-coder"]
MODEL_PATH = ""
for p in path_options:
    if os.path.exists(p):
        MODEL_PATH = os.path.abspath(p)
        break

if not MODEL_PATH:
    raise RuntimeError("❌ لم يتم العثور على مجلد الموديل.")

# 1. فحص العتاد
device_count = torch.cuda.device_count()
if device_count > 0:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
else:
    vram_gb = 0
    gpu_name = "CPU"

print(f"🚀 [ {PROJECT_NAME} ] يبدأ العمل...")
print(f"🖥️ الجهاز: {gpu_name} | CUDA: {device_count} | VRAM: {vram_gb:.1f} GB")

# 2. استراتيجية التحميل
loading_kwargs = {"device_map": "auto", "trust_remote_code": True}
if device_count == 0:
    loading_kwargs = {"device_map": {"": "cpu"}, "torch_dtype": torch.float32, "trust_remote_code": True}
elif vram_gb > 20:
    loading_kwargs["torch_dtype"] = torch.bfloat16
else:
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )

# 3. تحميل الموديل والـ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)

stop_tokens = ["<|im_end|>", "<|endoftext|>", "###"]

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.json"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
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
        user_data["conversations"][conv_id] = {
            "created_at": datetime.now().isoformat(),
            "title": user_text[:50] if user_text else "محادثة",
            "messages": []
        }
    
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
        import zipfile
        import re

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


@app.post("/v1/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    ip = _get_ip(request)
    saved = []
    for f in files:
        raw = await f.read()
        text = _extract_text(f, raw)
        path = UPLOAD_DIR / f"{ip.replace(':','_')}_{f.filename}"
        try:
            path.write_bytes(raw)
        except Exception:
            pass
        saved.append({
            "name": f.filename,
            "size": len(raw),
            "content": text[:12000]
        })
    return {"files": saved}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    conv_id = data.get("conversation_id")
    
    # تنظيف سريع
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list): content = " ".join([str(i) for i in content])
        cleaned.append({"role": m.get("role"), "content": content})

    user_text = cleaned[-1].get("content", "") if cleaned else ""
    input_text = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    ip = _get_ip(request)

    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=2048, temperature=0.7, do_sample=True)
    Thread(target=model.generate, kwargs=generation_kwargs).start()

    full_text = ""
    new_conv_id = conv_id

    async def generate():
        nonlocal full_text, new_conv_id
        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"
        for text in streamer:
            if await request.is_disconnected():
                break
            if not text.strip():
                continue
            full_text += text
            if any(s in text for s in stop_tokens):
                break
            yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
        new_conv_id = append_exchange(ip, user_text, full_text, conv_id)
        yield f"data: {json.dumps({'conversation_id': new_conv_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    print(f"✅ [ {PROJECT_NAME} ] جاهز!")
    print(f"🌐 واجهة الويب متاحة على: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")