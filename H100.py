import argparse
import json
import time
import uuid
import sys
import gc
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# اسم الموديل الافتراضي
MODEL_NAME = "models--Qwen--Qwen2.5-Coder-7B-Instruct"

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """سيرفر متعدد الخيوط لضمان استجابة فورية وعدم تعليق واجهة VS Code"""
    daemon_threads = True
    allow_reuse_address = True

def resolve_model_path(model_path: str, model_name: str | None) -> str:
    """تحديد مسار الموديل في جهازك الأوفلاين"""
    base = Path(model_path)
    if base.is_dir() and base.name == "models_cache":
        if model_name:
            candidate = base / model_name
            if candidate.is_dir(): base = candidate
            else:
                raw_candidate = base / f"models--{model_name}"
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
    print(f"--- 🚀 Loading H100 BEAST MODE (Full Precision): {model_path} ---")

    # 1. تحميل التوكنايزر
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        local_files_only=True, 
        fix_mistral_regex=True
    )
    # تفعيل سياق 32k للاستفادة من ذاكرة H100 الضخمة
    tokenizer.model_max_length = 32768 

    # 2. تحميل الموديل بكامل قوته (بدون ضغط)
    # نستخدم bfloat16 وهي الصيغة الأصلية والأسرع لـ H100
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # السرعة القصوى للـ H100
        device_map="auto",           # توزيع الموديل تلقائياً على الكروت المتوفرة
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    model.eval() 

    def _process_raw_messages(messages):
        """
        تمرير الرسائل كما هي (Passthrough) ليعتمد الموديل على تعليمات Cline،
        مع حذف التلقين المسبق (Pre-fill) فقط لمنع الهلوسة.
        """
        processed_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                content = "\n".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
            
            processed_messages.append({"role": role, "content": str(content)})

        # حذف التلقين المسبق الإجباري من Cline (آخر رسالة Assistant)
        if processed_messages and processed_messages[-1]['role'] == 'assistant':
             last_content = processed_messages[-1]['content'].strip()
             if not last_content or "<task" in last_content or "Here is" in last_content:
                print("ℹ️  Dropped Cline's forced pre-fill.")
                processed_messages.pop()

        return processed_messages

    class Handler(BaseHTTPRequestHandler):
        def _send_sse(self, payload):
            try:
                data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    
                    # استقبال الإعدادات من Cline (الحرارة، الحد الأقصى)
                    req_temp = float(body.get("temperature", 0.1))
                    req_max_tokens = int(body.get("max_tokens", 4096)) # رفعنا الحد الافتراضي لـ 4096 لأن الـ H100 يتحمل

                    print(f"📥 Request: Temp={req_temp}, MaxTokens={req_max_tokens}...")
                    
                    final_messages = _process_raw_messages(body.get("messages", []))
                    
                    prompt = tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    gen_kwargs = dict(
                        inputs, 
                        streamer=streamer, 
                        max_new_tokens=req_max_tokens,
                        temperature=req_temp,
                        do_sample=True 
                    )
                    
                    thread = Thread(target=model.generate, kwargs=gen_kwargs)
                    thread.start()

                    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
                    for new_text in streamer:
                        if not self._send_sse({
                            "id": chat_id,
                            "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}]
                        }):
                            break 
                    
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except: pass
                    print(f"✅ Finished.")
                    
                    # لا نحتاج لتنظيف الذاكرة بعنف مع H100، الذاكرة ضخمة جداً
                    # gc.collect() 

                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    try: self.send_error(500)
                    except: pass

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": args.model_name}]}).encode())

    print(f"--- ✅ H100 Server Running at http://{args.host}:{args.port}/v1 ---")
    try:
        httpd = ThreadedHTTPServer((args.host, args.port), Handler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()