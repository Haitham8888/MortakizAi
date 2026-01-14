import torch
import uvicorn
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# --- إعداد المسار المطلق لتفادي خطأ HFValidationError ---
# جرب المسار الجديد أولاً، إذا لم يوجد جرب القديم
path_options = [
    "./models_cache/models--Qwen--Qwen2.5-Coder-7B-Instruct",
    "./qwen-coder"
]

MODEL_PATH = ""
for p in path_options:
    if os.path.exists(p):
        MODEL_PATH = os.path.abspath(p)
        break

if not MODEL_PATH:
    raise RuntimeError("❌ لم يتم العثور على مجلد الموديل في أي من المسارات المتوقعة!")

# 1. فحص مواصفات الكرت
device_count = torch.cuda.device_count()
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
gpu_name = torch.cuda.get_device_name(0)

print(f"🚀 [ مَرْتَكَز - MortakizAi ] يبدأ العمل الآن...")
print(f"🖥️ العتاد المكتشف: {gpu_name} ({vram_gb:.2f} GB VRAM)")

# 2. استراتيجية التشغيل
loading_kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
}

if vram_gb > 20:
    print("🔥 نمط الأداء الأقصى (BF16)")
    loading_kwargs["torch_dtype"] = torch.bfloat16
else:
    print("💡 نمط الحفاظ على الذاكرة (4-bit)")
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

# 3. تحميل الموديل والـ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)

app = FastAPI()

def clean_messages(messages):
    cleaned = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
        cleaned.append({"role": role, "content": content})
    return cleaned

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = clean_messages(data.get("messages", []))
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=4096, temperature=0.7, do_sample=True)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def generate_chunks():
        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'index': 0}]})}\n\n"
        for new_text in streamer:
            if await request.is_disconnected(): break
            if new_text:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': new_text}, 'index': 0}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_chunks(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")