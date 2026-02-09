import argparse
import json
import uuid
import sys
import torch
from threading import Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

MODEL_NAME = "models--Qwen--Qwen2.5-Coder-7B-Instruct"

# --- إعدادات A4000 ---
MAX_INPUT_TOKENS = 8000    # يمكننا رفع السياق الآن لأن الموديل مضغوط!
MAX_NEW_TOKENS_CAP = 1024 
# ---------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, args.model_name)
    print(f"--- 🚀 A4000 MODE (4-bit Quantization): Loading from {model_path} ---")

    # إعدادات الضغط لتقليص الحجم من 14 جيجا إلى 5 جيجا
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 1. التوكنايزر
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True,
            fix_mistral_regex=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # 2. الموديل (مضغوط 4-bit)
    print("--- ⏳ Loading Model (Compressed for 16GB VRAM)... ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config, # <--- سر العمل على A4000
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2", # لا يزال يعمل بسرعة عالية
            low_cpu_mem_usage=True
        )
        print("--- ✅ Model Loaded in 4-bit (High Speed / Low Memory) ---")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # لا حاجة لـ model.eval() مع bitsandbytes لأنه يضبطها تلقائياً

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
                    
                    req_max = int(body.get("max_tokens", 512))
                    final_max_new_tokens = min(req_max, MAX_NEW_TOKENS_CAP)

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
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    input_len = inputs.input_ids.shape[1]
                    
                    if input_len > MAX_INPUT_TOKENS:
                        inputs.input_ids = inputs.input_ids[:, -MAX_INPUT_TOKENS:]
                        inputs.attention_mask = inputs.attention_mask[:, -MAX_INPUT_TOKENS:]

                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    
                    gen_kwargs = dict(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        streamer=streamer,
                        max_new_tokens=final_max_new_tokens,
                        do_sample=False,
                        use_cache=True    
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
                    print(f"⚠️ Request Error: {e}")
                    self.send_error(500, str(e))

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": args.model_name}]}).encode())

    print(f"--- 🚀 A4000 Server Ready at http://{args.host}:{args.port}/v1 ---")
    httpd = ThreadedHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()