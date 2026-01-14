import torch
import uvicorn
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
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
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
gpu_name = torch.cuda.get_device_name(0)

print(f"🚀 [ {PROJECT_NAME} ] يبدأ العمل...")

# 2. استراتيجية التحميل
loading_kwargs = {"device_map": "auto", "trust_remote_code": True}
if vram_gb > 20:
    loading_kwargs["torch_dtype"] = torch.bfloat16
else:
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )

# 3. تحميل الموديل والـ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)

stop_tokens = ["<|im_end|>", "<|endoftext|>", "###"]

app = FastAPI()

# --- واجهة الويب (HTML UI) ---
CHAT_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MortakizAi | مَرْتَكَز</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .chat-container { height: 75vh; overflow-y: auto; scrollbar-width: thin; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 15px; margin-bottom: 10px; }
        .user-msg { background-color: #1e293b; align-self: flex-start; margin-right: auto; }
        .ai-msg { background-color: #334155; align-self: flex-end; margin-left: auto; border-left: 4px solid #38bdf8; }
        pre { background: #000; padding: 10px; border-radius: 8px; overflow-x: auto; color: #10b981; direction: ltr; text-align: left; }
    </style>
</head>
<body class="flex flex-col min-h-screen">
    <header class="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-900">
        <h1 class="text-xl font-bold text-sky-400">🛡️ مَرْتَكَز - MortakizAi</h1>
        <span class="text-xs bg-sky-900 text-sky-200 px-2 py-1 rounded">متصل محلياً</span>
    </header>

    <main class="flex-grow container mx-auto p-4 max-w-4xl">
        <div id="chatBox" class="chat-container flex flex-col space-y-4 p-2">
            <div class="message ai-msg">أهلاً بك يا هيثم! أنا "مَرْتَكَز"، ذكاؤك الاصطناعي المحلي. كيف يمكنني مساعدتك في الكود اليوم؟</div>
        </div>
    </main>

    <footer class="p-4 bg-slate-900 border-t border-slate-700">
        <div class="container mx-auto max-w-4xl flex gap-2">
            <input type="text" id="userInput" placeholder="اكتب سؤالك هنا..." class="w-full p-3 bg-slate-800 border border-slate-600 rounded-lg focus:outline-none focus:border-sky-500">
            <button id="sendBtn" class="bg-sky-600 hover:bg-sky-500 px-6 py-2 rounded-lg font-bold transition">إرسال</button>
        </div>
    </footer>

    <script>
        const chatBox = document.getElementById('chatBox');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(text, isAi) {
            const div = document.createElement('div');
            div.className = `message ${isAi ? 'ai-msg' : 'user-msg'}`;
            div.innerText = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            
            userInput.value = '';
            addMessage(text, false);
            
            const aiDiv = addMessage("...", true);
            let fullText = "";

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: [{ role: 'user', content: text }] })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                aiDiv.innerText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                            const data = JSON.parse(line.substring(6));
                            const content = data.choices[0].delta.content || "";
                            fullText += content;
                            aiDiv.innerText = fullText;
                        }
                    }
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            } catch (err) {
                aiDiv.innerText = "⚠️ خطأ في الاتصال بالسيرفر.";
            }
        }

        sendBtn.onclick = sendMessage;
        userInput.onkeypress = (e) => { if(e.key === 'Enter') sendMessage(); };
    </script>
</body>
</html>
"""

# --- نقاط النهاية (API Endpoints) ---

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return CHAT_HTML

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    # تنظيف سريع
    cleaned = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list): content = " ".join([str(i) for i in content])
        cleaned.append({"role": m.get("role"), "content": content})

    input_text = tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=2048, temperature=0.7, do_sample=True)
    Thread(target=model.generate, kwargs=generation_kwargs).start()

    async def generate():
        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})}\\n\\n"
        for text in streamer:
            if await request.is_disconnected(): break
            if any(s in text for s in stop_tokens): break
            yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    print(f"✅ [ {PROJECT_NAME} ] جاهز!")
    print(f"🌐 واجهة الويب متاحة على: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")