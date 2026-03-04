import sys
import subprocess
from pathlib import Path

def find_model_path(base_dir):
    """يبحث عن ملف config.json لاستخراج المسار النهائي للنموذج"""
    print(f"🔍 جاري البحث عن النموذج داخل: {base_dir} ...")
    for path in Path(base_dir).rglob('config.json'):
        return str(path.parent)
    return None

def main():
    if len(sys.argv) < 2:
        print("❌ خطأ: يرجى تمرير المسار الرئيسي للمجلد الذي يحتوي على النموذج.")
        print("💡 مثال: python run_ai.py /path/to/models--Qwen--Qwen3-Coder-30B-A3B-Instruct [served_model_name] [max_model_len] [gpu_memory_utilization]")
        sys.exit(1)

    base_dir = sys.argv[1]
    served_model_name = sys.argv[2] if len(sys.argv) > 2 else "q3"
    max_model_len = sys.argv[3] if len(sys.argv) > 3 else "32768" # زيادة طول السياق لدعم المشاريع الكبيرة
    gpu_memory_utilization = sys.argv[4] if len(sys.argv) > 4 else "0.9" # زيادة الاستفادة من الذاكرة قليلاً
    
    model_dir = find_model_path(base_dir)

    if not model_dir:
        print("❌ خطأ: لم يتم العثور على المسار الصحيح (تأكد من وجود ملف config.json).")
        sys.exit(1)

    print(f"✅ تم العثور على المسار الصحيح: {model_dir}")
    print(f"🏷️ الاسم المستعار: {served_model_name}")
    print(f"� طول السياق (Context Length): {max_model_len}")
    print(f"📈 استخدام ذاكرة GPU: {gpu_memory_utilization}")
    print("�🚀 جاري تشغيل خادم vLLM... (لإيقاف الخادم اضغط Ctrl+C)")

    # إعداد أمر التشغيل
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", served_model_name,
        "--host", "0.0.0.0",
        "--port", "8888",
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_memory_utilization
    ]

    try:
        # تشغيل الخادم
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف الخادم بنجاح.")

if __name__ == "__main__":
    main()
