import argparse
import json
import time
import uuid
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# الموديل الموصى به للبرمجة الاحترافية
MODEL_NAME = "models--Qwen--Qwen2.5-Coder-7B-Instruct"

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """تسمح بمعالجة طلبات متعددة لضمان استجابة واجهة VS Code السريعة"""
    daemon_threads = True

def resolve_model_path(model_path: str, model_name: str | None) -> str:
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
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, args.model_name)
    print(f"--- 🚀 Loading Professional Model: {model_path} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    
    # تحميل الموديل بذكاء (توزيع تلقائي واستخدام bfloat16 لسرعة قصوى)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    def _normalize_messages(messages):
        """تصفية تعليمات Cline الضخمة لتركيز ذكاء الموديل على الرد البرمجي فقط"""
        normalized = [
            {
                "role": "system",
                "content": "You are a professional software engineer. Provide clear, concise answers. If a tool is needed, use the requested XML format exactly. Stop immediately after answering."
            }
        ]
        # نأخذ السياق الضروري فقط لتجنب تشتت الموديل
        for msg in messages or []:
            role = msg.get("role", "user")
            if role == "system": continue 
            
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
            
            normalized.append({"role": role, "content": str(content)})
        return normalized

    class Handler(BaseHTTPRequestHandler):
        def _send_sse(self, payload):
            try:
                data = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except:
                return False

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                
                print(f"📥 Generating code using Qwen 7B...")
                
                prompt = tokenizer.apply_chat_template(_normalize_messages(body.get("messages", [])), tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                # إضافة علامات التوقف (Stop Sequences) لمنع التكرار
                gen_kwargs = dict(
                    inputs, 
                    streamer=streamer, 
                    max_new_tokens=2048, 
                    temperature=0.1, 
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                chat_id = f"chatcmpl-{uuid.uuid4().hex}"
                for new_text in streamer:
                    # التحقق من أن الموديل لم يبدأ في تكرار تعليمات النظام
                    if "system" in new_text.lower() or "user <task>" in new_text.lower():
                        break
                    if not self._send_sse({
                        "id": chat_id,
                        "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}]
                    }):
                        break
                
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except: pass
                print(f"✅ Finished Task.")

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": args.model_name}]}).encode())

    print(f"--- ✅ Professional Server running at http://0.0.0.0:{args.port}/v1 ---")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()