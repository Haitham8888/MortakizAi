import argparse
import json
import uuid
import sys
import torch
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- 💎 Qwen3 MoE (الوحش الجديد) ---
# تأكد أنك حملت الموديل بهذا الاسم في مجلد models_cache
MODEL_NAME = "models--Qwen--Qwen2.5-Coder-7B-Instruct"

# --- ⚡ إعدادات السرعة والذاكرة ---
MAX_INPUT_TOKENS = 32000   # MoE يدعم سياق ضخم
MAX_NEW_TOKENS_CAP = 4096  # نسمح له بالكتابة بحرية
# ----------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def resolve_model_path(model_path: str, model_name: str | None) -> str:
    """البحث عن الموديل محلياً"""
    # إذا كان الاسم مساراً مباشراً لموديل HF
    if "/" in model_name: return model_name
    
    # البحث في الكاش المحلي
    base = Path(model_path)
    if base.is_dir() and base.name == "models_cache":
        if model_name:
            candidate = base / model_name
            if candidate.is_dir(): base = candidate
            else:
                raw_candidate = base / f"models--{model_name.replace('/', '--')}"
                if raw_candidate.is_dir(): base = raw_candidate
    snapshots_dir = base / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshots:
            return str(max(snapshots, key=lambda p: p.stat().st_mtime))
    return str(base)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models_cache")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, args.model_name)
    print(f"--- 🚀 MULTI-GPU Qwen3 MoE SERVER: {model_path} ---")

    # 1. تحميل التوكنايزر
    print("--- ⏳ Loading Tokenizer... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # 2. تحميل الموديل (توزيع تلقائي على الكروت)
    print("--- ⏳ Loading MoE Model (Auto-Distributing across GPUs)... ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",       # Qwen3 يفضل auto لاختيار الأنسب
            device_map="auto",        # <--- السر هنا: يوزع الحمل على كل الكروت
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2" # سرعة H100
        )
        print("--- ✅ Qwen3 MoE Loaded & Distributed! ---")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # طباعة خريطة التوزيع لنرى كيف تم تقسيم الموديل
    if hasattr(model, "hf_device_map"):
        print(f"ℹ️ GPU Map: {json.dumps(model.hf_device_map, indent=2)}")

    model.eval()

    class Handler(BaseHTTPRequestHandler):
        def _send_sse(self, payload):
            try:
                data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except: return False

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    
                    req_max = int(body.get("max_tokens", 4096))
                    final_max_new_tokens = min(req_max, MAX_NEW_TOKENS_CAP)
                    
                    # بارامترات Qwen3 الخاصة (Sampling)
                    req_temp = float(body.get("temperature", 0.7))
                    req_top_p = float(body.get("top_p", 0.8))

                    messages = body.get("messages", [])
                    clean_msgs = []
                    for m in messages:
                        content = m.get("content", "")
                        if isinstance(content, list): 
                            content = "\n".join([str(x) for x in content])
                        clean_msgs.append({"role": m.get("role"), "content": str(content)})
                    
                    if clean_msgs and clean_msgs[-1]['role'] == 'assistant':
                        clean_msgs.pop()

                    prompt = tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)
                    
                    # نقل البيانات للـ GPU (accelerate سيتولى الباقي)
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    
                    # Truncation
                    if inputs.input_ids.shape[1] > MAX_INPUT_TOKENS:
                        inputs.input_ids = inputs.input_ids[:, -MAX_INPUT_TOKENS:]
                        inputs.attention_mask = inputs.attention_mask[:, -MAX_INPUT_TOKENS:]

                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    gen_kwargs = dict(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        streamer=streamer,
                        max_new_tokens=final_max_new_tokens,
                        temperature=req_temp, 
                        top_p=req_top_p,
                        top_k=20,               # إعدادات Qwen3
                        repetition_penalty=1.05,# إعدادات Qwen3
                        do_sample=True,         # ضروري لـ MoE للإبداع
                    )
                    
                    thread = Thread(target=model.generate, kwargs=gen_kwargs)
                    thread.start()

                    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
                    for new_text in streamer:
                        if not self._send_sse({
                            "id": chat_id,
                            "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}]
                        }): break 
                    
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    
                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    self.send_error(500, str(e))

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": args.model_name}]}).encode())

    print(f"--- 🚀 Qwen3 Multi-GPU Server Ready at http://{args.host}:{args.port}/v1 ---")
    httpd = ThreadedHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()