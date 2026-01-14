import torch
import uvicorn
import json
import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

# --- إعدادات الهوية والمسارات ---
PROJECT_NAME = "مَرْتَكَز - MortakizAi"
# الكود سيبحث في المسارين المحتملين ويختار الموجود منهما
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
    raise RuntimeError("❌ خطأ: لم يتم العثور على مجلد الموديل. تأكد من وجوده بجانب الملف.")

# 1. فحص العتاد (GPU Detection)
device_count = torch.cuda.device_count()
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
gpu_name = torch.cuda.get_device_name(0)

print(f"🚀 [ {PROJECT_NAME} ] يبدأ العمل الآن...")
print(f"🖥️ العتاد المكتشف: {gpu_name} ({vram_gb:.2f} GB VRAM)")

# 2. استراتيجية التحميل (Adaptive Strategy)
loading_kwargs = {
    "device_map": "auto",
    "trust_remote_code": True,
}

if vram_gb > 20:
    print("🔥 نمط الأداء الأقصى مفعل (Full BF16)...")
    loading_kwargs["torch_dtype"] = torch.bfloat16
    try:
        import flash_attn
        loading_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        loading_kwargs["attn_implementation"] = "sdpa"
else:
    print("💡 نمط الحفاظ على الذاكرة مفعل (4-bit)...")
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    loading_kwargs["attn_implementation"] = "sdpa"

# 3. تحميل الـ Tokenizer والموديل
print(f"📦 جاري شحن ملفات الموديل من المسار المطلق...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)

# تحديد رموز التوقف لمنع الهذيان والتكرار
stop_tokens = ["<|im_end|>", "<|endoftext|>", "###", "Instruction:", "Response:"]

app = FastAPI()

def clean_messages(messages):
    """تنظيف وتوحيد مدخلات Cline"""
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
    
    # تنسيق المدخلات حسب قالب Qwen الرسمي
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # إعدادات التوليد المحسنة للأداء
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=4096,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1, # يمنع تكرار نفس الجمل
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # بدء المعالجة في خيط منفصل
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def generate_chunks():
        try:
            # إرسال كائن الاستجابة الأولي لفتح القناة مع Cline
            yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'index': 0}]})}\n\n"
            
            for new_text in streamer:
                # التحقق إذا أغلق المستخدم النافذة في VSCode
                if await request.is_disconnected():
                    print("🔌 تم قطع الاتصال.. إيقاف التوليد.")
                    break
                
                # التحقق من كلمات التوقف يدوياً لزيادة الدقة
                if any(stop_word in new_text for stop_word in stop_tokens):
                    break

                if new_text:
                    chunk = {"choices": [{"delta": {"content": new_text}, "index": 0}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"⚠️ تنبيه بسيط: {e}")

    return StreamingResponse(generate_chunks(), media_type="text/event-stream")

if __name__ == "__main__":
    mode_info = "High-Power" if vram_gb > 20 else "Efficient 4-bit"
    print(f"✅ [ {PROJECT_NAME} ] جاهز الآن على http://localhost:8000")
    print(f"⚙️ النمط النشط: {mode_info}")
    # log_level='error' لإبقاء التيرمينال نظيفاً من طلبات الـ HTTP
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")