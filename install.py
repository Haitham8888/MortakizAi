import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# الإعدادات
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# تحويل اسم الموديل للتنسيق الذي طلبته (models--author--modelname)
formatted_name = "models--" + MODEL_ID.replace("/", "--")
BASE_CACHE_DIR = "./models_cache"
FINAL_PATH = os.path.join(BASE_CACHE_DIR, formatted_name)

# إنشاء المجلدات
os.makedirs(FINAL_PATH, exist_ok=True)

print(f"⏳ جاري التحميل في المسار المخصص:")
print(f"📂 {FINAL_PATH}")

# 1. تحميل الـ Tokenizer وحفظه
print("1️⃣ جاري تحميل الـ Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(FINAL_PATH)

# 2. تحميل الموديل بدقة BF16 (مناسب لـ H100)
print("2️⃣ جاري تحميل الموديل (سيتم استخدام الـ RAM للتحميل لتوفير الـ VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 3. حفظ الموديل في المسار النهائي
print(f"3️⃣ جاري حفظ الملفات في المجلد النهائي...")
model.save_pretrained(FINAL_PATH)

print("-" * 30)
print(f"✅ تم الحفظ بنجاح في:")
print(f"📍 {os.path.abspath(FINAL_PATH)}")