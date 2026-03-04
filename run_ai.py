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
        print("💡 مثال: python run_ai.py /path/to/models--Qwen--Qwen3-Coder-30B-A3B-Instruct")
        sys.exit(1)

    base_dir = sys.argv[1]
    model_dir = find_model_path(base_dir)

    if not model_dir:
        print("❌ خطأ: لم يتم العثور على المسار الصحيح (تأكد من وجود ملف config.json).")
        sys.exit(1)

    print(f"✅ تم العثور على المسار الصحيح: {model_dir}")
    print("🚀 جاري تشغيل خادم vLLM... (لإيقاف الخادم وتحرير الذاكرة، اضغط Ctrl+C)")

    # إعداد أمر التشغيل
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--host", "0.0.0.0",
        "--port", "8888",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.8"
    ]

    try:
        # تشغيل الخادم
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف الخادم بنجاح. ذاكرة H100 الآن حرة بالكامل لمشاريعك الأخرى!")

if __name__ == "__main__":
    main()
