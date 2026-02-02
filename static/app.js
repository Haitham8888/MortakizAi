// ===== DOM Elements =====
const chatBox = document.getElementById('chatBox');
const hero = document.getElementById('hero');
const chatPanel = document.getElementById('chatPanel');
const langToggle = document.getElementById('langToggle');
const titleText = document.getElementById('titleText');
const subtitleText = document.getElementById('subtitleText');
const statusText = document.getElementById('statusText');

// Sidebar elements
const sidebar = document.getElementById('sidebar');
const newChatBtn = document.getElementById('newChatBtn');
const chatsList = document.getElementById('chatsList');
const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
const openSidebarBtn = document.getElementById('openSidebarBtn');
const clearAllChatsBtn = document.getElementById('clearAllChatsBtn');
const newChatText = document.getElementById('newChatText');
const chatsLabel = document.getElementById('chatsLabel');

// Hero inputs (initial state)
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const uploadBtn = document.getElementById('uploadBtn');

// Chat inputs (after first message)
const userInputChat = document.getElementById('userInputChat');
const sendBtnChat = document.getElementById('sendBtnChat');
const uploadBtnChat = document.getElementById('uploadBtnChat');

const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.multiple = true;
fileInput.accept = '.txt,.md,.py,.js,.ts,.json,.yaml,.yml,.html,.css,.cpp,.c,.h,.java,.cs,.go,.rs,.php,.rb,.sh,.ps1,.sql,.xml,.docx';

// ===== Translations =====
const translations = {
    ar: {
        dir: 'rtl',
        lang: 'ar',
        title: 'مَرْتَكَز - MortakizAi',
        subtitle: 'مساعد البرمجة المحلي',
        status: 'متصل محلياً',
        panelTitle: 'المحادثة',
        panelSubtitle: 'اسأل عن الأكواد، التصحيحات، أو الاقتراحات',
        placeholder: 'اكتب سؤالك هنا...',
        send: 'إرسال',
        welcome: 'أهلاً بك! أنا "مَرْتَكَز"، ذكاؤك الاصطناعي المحلي. كيف يمكنني مساعدتك في الكود اليوم؟',
        error: '⚠️ خطأ في الاتصال بالسيرفر.',
        toggleLabel: 'EN',
        copy: 'نسخ الكود',
        copied: 'تم النسخ',
        newChat: 'محادثة جديدة',
        prevChats: 'المحادثات السابقة',
        noChats: 'لا توجد محادثات سابقة',
        heroTitle: 'ما الذي تريد البدء به؟',
        deleteAll: 'حذف جميع المحادثات',
        hideSidebar: 'إخفاء القائمة',
        openSidebar: 'فتح القائمة',
        delete: 'حذف',
        fileReadError: 'تعذر قراءة بعض الملفات. جرّب حفظها كنص عادي أو إعادة رفعها.'
    },
    en: {
        dir: 'ltr',
        lang: 'en',
        title: 'MortakizAi',
        subtitle: 'Local coding copilot',
        status: 'Local connection',
        panelTitle: 'Chat',
        panelSubtitle: 'Ask for code, fixes, or suggestions',
        placeholder: 'Type your question...',
        send: 'Send',
        welcome: 'Hi there! I\'m "Mortakiz", your local AI. How can I help with code today?',
        error: '⚠️ Connection error with server.',
        toggleLabel: 'عربي',
        copy: 'Copy code',
        copied: 'Copied!',
        newChat: 'New Chat',
        prevChats: 'Previous Chats',
        noChats: 'No previous chats',
        heroTitle: 'What would you like to start with?',
        deleteAll: 'Delete all chats',
        hideSidebar: 'Hide sidebar',
        openSidebar: 'Open sidebar',
        delete: 'Delete',
        fileReadError: 'Some files could not be read. Please save them as plain text and re-upload.'
    }
};

// ===== State Variables =====
let currentLang = 'ar';
let conversationStarted = false;
let attachments = [];
let currentConversationId = null;
let conversations = [];
let isGenerating = false;
let userScrolledUp = false;
let currentMessages = []; // مصفوفة لتخزين رسائل المحادثة الحالية
const SIDEBAR_KEY = 'mortakiz-sidebar';

// ===== Scroll Functions =====
function isNearBottom() {
    const threshold = 100;
    return chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight < threshold;
}

function scrollToBottom() {
    if (!userScrolledUp) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

chatBox.addEventListener('scroll', () => {
    if (isGenerating) {
        userScrolledUp = !isNearBottom();
    }
});

// ===== Helper Functions =====
function escapeHTML(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// دالة لإصلاح المسافات البادئة في الكود
function fixCodeIndentation(code, lang) {
    const lines = code.split('\n');
    if (lines.length < 2) return code;
    
    // أنماط الكود الشائعة التي تحتاج مسافات
    const indentPatterns = {
        'java': [
            { match: /^(public|private|protected)\s+(static\s+)?(class|interface|enum)\s+/, indent: 0 },
            { match: /^(public|private|protected)\s+(static\s+)?[\w<>\[\]]+\s+\w+\s*\(/, indent: 1 },
            { match: /^\}$/, indent: 0, dedent: true },
            { match: /^(if|else|for|while|switch|try|catch|finally)\s*[\(\{]?/, indent: 2 },
            { match: /^(return|System\.|print|throw|break|continue)/, indent: 2 },
            { match: /^\w+\s*[=;]/, indent: 2 }
        ],
        'javascript': [
            { match: /^(function|class|const|let|var)\s+\w+/, indent: 0 },
            { match: /^\}$/, indent: 0, dedent: true },
            { match: /^(if|else|for|while|switch|try|catch)\s*[\(\{]?/, indent: 1 },
            { match: /^(return|console\.|throw)/, indent: 1 }
        ],
        'python': [
            { match: /^(def|class|async def)\s+/, indent: 0 },
            { match: /^(if|elif|else|for|while|try|except|finally|with)[\s:]/, indent: 1 },
            { match: /^(return|print|raise|pass|break|continue)/, indent: 2 }
        ],
        'c': [
            { match: /^(int|void|char|float|double|struct)\s+\w+\s*\(/, indent: 0 },
            { match: /^\}$/, indent: 0, dedent: true },
            { match: /^(if|else|for|while|switch)\s*\(/, indent: 1 },
            { match: /^(return|printf|break|continue)/, indent: 2 }
        ]
    };
    
    // تطبيق الإصلاح فقط إذا كان الكود بدون مسافات
    const hasIndentation = lines.some(line => line.match(/^\s{2,}/));
    if (hasIndentation) return code;
    
    const langKey = (lang || '').toLowerCase();
    const patterns = indentPatterns[langKey] || indentPatterns['java'] || [];
    
    let braceLevel = 0;
    const fixedLines = lines.map(line => {
        const trimmed = line.trim();
        if (!trimmed) return '';
        
        // تتبع الأقواس المعقوفة
        if (trimmed === '}') {
            braceLevel = Math.max(0, braceLevel - 1);
        }
        
        const indent = '    '.repeat(braceLevel);
        const result = indent + trimmed;
        
        // زيادة المستوى بعد فتح القوس
        if (trimmed.endsWith('{')) {
            braceLevel++;
        }
        
        return result;
    });
    
    return fixedLines.join('\n');
}

function formatMarkdown(text) {
    // Tables: convert markdown tables to HTML
    text = text.replace(/^(\|.+\|)\r?\n(\|[-:| ]+\|)\r?\n((?:\|.+\|\r?\n?)+)/gm, (match, header, separator, body) => {
        const headers = header.split('|').filter(h => h.trim()).map(h => `<th>${h.trim()}</th>`).join('');
        const rows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<div class="table-wrapper"><table class="md-table"><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
    });
    
    // Bold: **text** or __text__
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    // Italic: *text* or _text_
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/_([^_]+)_/g, '<em>$1</em>');
    // Inline code: `code`
    text = text.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
    // Line breaks - single only, trim multiple
    text = text.replace(/\n{2,}/g, '\n');
    text = text.replace(/\n/g, '<br>');
    // Remove leading/trailing br tags
    text = text.replace(/^(<br>)+/g, '');
    text = text.replace(/(<br>)+$/g, '');
    return text;
}

function renderContent(raw, el) {
    const parts = raw.split(/```/);
    let html = '';

    parts.forEach((part, idx) => {
        if (idx % 2 === 0) {
            const escaped = escapeHTML(part);
            const formatted = formatMarkdown(escaped);
            if (formatted.trim().length) html += `<div class="msg-text" dir="auto">${formatted}</div>`;
        } else {
            // Extract language from first line if present
            const lines = part.split('\n');
            let lang = '';
            let code = part;
            
            if (lines[0] && /^[a-zA-Z0-9_+-]+$/.test(lines[0].trim())) {
                lang = lines[0].trim().toLowerCase();
                // الحفاظ على المسافات الأصلية - فقط إزالة السطر الأول (اللغة)
                code = lines.slice(1).join('\n');
                // إزالة سطر فارغ واحد من البداية والنهاية فقط
                if (code.startsWith('\n')) code = code.substring(1);
                if (code.endsWith('\n')) code = code.substring(0, code.length - 1);
            } else {
                // إذا لم يكن هناك لغة، إزالة سطر فارغ من البداية والنهاية فقط
                if (code.startsWith('\n')) code = code.substring(1);
                if (code.endsWith('\n')) code = code.substring(0, code.length - 1);
            }
            // إزالة أي أسطر فارغة متكررة في البداية والنهاية
            code = code.replace(/^\s*\n+/, '').replace(/\n+\s*$/, '');
            
            // إصلاح المسافات البادئة إذا كانت مفقودة
            code = fixCodeIndentation(code, lang);
            
            html += `
                <div class="code-block">
                    <div class="code-block-header">
                        <span class="code-block-lang">${lang || 'code'}</span>
                        <button class="copy-btn" data-copy="">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>نسخ الكود</span>
                        </button>
                    </div>
                    <pre><code>${escapeHTML(code)}</code></pre>
                </div>
            `;
        }
    });

    el.innerHTML = html || '<div class="msg-text" dir="auto">…</div>';

    el.querySelectorAll('[data-copy]').forEach(btn => {
        btn.onclick = () => {
            const codeBlock = btn.closest('.code-block');
            const codeEl = codeBlock?.querySelector('pre code');
            if (!codeEl) return;
            
            const checkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>`;
            const copyIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
            
            navigator.clipboard.writeText(codeEl.textContent).then(() => {
                btn.innerHTML = `${checkIcon}<span>${translations[currentLang].copied}</span>`;
                setTimeout(() => {
                    btn.innerHTML = `${copyIcon}<span>${translations[currentLang].copy}</span>`;
                }, 2000);
            });
        };
    });
}

function addMessage(text, isAi, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${isAi ? 'ai-msg' : 'user-msg'}`;
    if (isHtml) {
        div.innerHTML = `<div class="msg-text" dir="auto">${text}</div>`;
    } else {
        renderContent(text, div);
    }
    chatBox.appendChild(div);
    scrollToBottom();
    updateHeroState();
    return div;
}

function updateHeroState() {
    // Not needed anymore
}

function revealChat() {
    hero.classList.add('hidden');
    chatPanel.classList.remove('hidden');
    userInputChat.focus();
}

function showHero() {
    hero.classList.remove('hidden');
    chatPanel.classList.add('hidden');
    chatBox.innerHTML = '';
    userInput.focus();
}

// ===== Sidebar Functions =====
function toggleSidebar(show) {
    const isOpen = show !== undefined ? show : sidebar.classList.contains('sidebar--collapsed');
    sidebar.classList.toggle('sidebar--collapsed', !isOpen);
    openSidebarBtn.classList.toggle('hidden', isOpen);
    localStorage.setItem(SIDEBAR_KEY, isOpen ? 'open' : 'closed');
}

(function initSidebar() {
    const saved = localStorage.getItem(SIDEBAR_KEY);
    if (saved === 'closed') {
        toggleSidebar(false);
    }
})();

toggleSidebarBtn.addEventListener('click', () => toggleSidebar(false));
openSidebarBtn.addEventListener('click', () => toggleSidebar(true));

async function loadConversations() {
    try {
        const res = await fetch('/v1/conversations');
        if (!res.ok) return;
        const data = await res.json();
        conversations = data.conversations || [];
        renderConversationsList();
    } catch (e) {
        // ignore
    }
}

function renderConversationsList() {
    chatsList.innerHTML = '';
    if (conversations.length === 0) {
        chatsList.innerHTML = `<p class="no-chats">${translations[currentLang].noChats}</p>`;
        return;
    }
    conversations.forEach(conv => {
        const item = document.createElement('button');
        item.className = `chat-item ${conv.id === currentConversationId ? 'chat-item--active' : ''}`;
        // تنظيف العنوان - إزالة الأسطر الجديدة وقصه
        let title = (conv.title || (currentLang === 'ar' ? 'محادثة' : 'Chat')).replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
        if (title.length > 35) title = title.substring(0, 35) + '...';
        item.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span class="chat-item__title">${title}</span>
            <button class="chat-item__delete" data-conv-id="${conv.id}" title="${translations[currentLang].delete}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
            </button>
        `;
        item.addEventListener('click', (e) => {
            if (!e.target.closest('.chat-item__delete')) {
                loadConversation(conv.id);
            }
        });
        chatsList.appendChild(item);
    });
    
    // Delete buttons
    chatsList.querySelectorAll('.chat-item__delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const convId = btn.dataset.convId;
            await deleteConversation(convId);
        });
    });
}

// دالة لاستخراج نص العرض من المحتوى الكامل (إخفاء محتوى الملفات)
function extractDisplayText(content, isUser) {
    if (!isUser) return content;
    
    // البحث عن نمط الملفات: "ملف: اسم_الملف\nمحتوى..."
    const filePattern = /\n\nملف: ([^\n]+)\n[\s\S]*$/;
    const match = content.match(filePattern);
    
    if (match) {
        // استخراج النص قبل الملفات
        const textOnly = content.replace(filePattern, '').trim();
        // استخراج أسماء الملفات
        const fileMatches = content.match(/ملف: ([^\n]+)/g) || [];
        const fileNames = fileMatches.map(f => f.replace('ملف: ', ''));
        
        if (fileNames.length > 0) {
            const fileTags = fileNames.map(name => 
                `<span class="file-tag"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>${name}</span>`
            ).join(' ');
            return textOnly ? `${textOnly}<div class="file-tags">${fileTags}</div>` : `<div class="file-tags">${fileTags}</div>`;
        }
    }
    return content;
}

async function loadConversation(convId) {
    try {
        const res = await fetch(`/v1/conversations/${convId}`);
        if (!res.ok) return;
        const data = await res.json();
        currentConversationId = convId;
        chatBox.innerHTML = '';
        currentMessages = []; // إعادة تعيين المصفوفة
        revealChat();
        (data.messages || []).forEach(msg => {
            const isUser = msg.role === 'user';
            const displayContent = extractDisplayText(msg.content || '', isUser);
            const hasHtml = displayContent.includes('<div class="file-tags">');
            addMessage(displayContent, !isUser, hasHtml);
            // تخزين الرسالة الكاملة في المصفوفة للسياق
            currentMessages.push({
                role: msg.role,
                content: msg.content || ''
            });
        });
        renderConversationsList();
    } catch (e) {
        // ignore
    }
}

async function deleteConversation(convId) {
    try {
        await fetch(`/v1/conversations/${convId}`, { method: 'DELETE' });
        if (currentConversationId === convId) {
            currentConversationId = null;
            showHero();
        }
        await loadConversations();
    } catch (e) {
        // ignore
    }
}

async function clearAllConversations() {
    console.log('clearAllConversations called');
    const msg = currentLang === 'ar' ? 'هل أنت متأكد من حذف جميع المحادثات؟' : 'Are you sure you want to delete all conversations?';
    if (!confirm(msg)) {
        return;
    }
    try {
        const resp = await fetch('/v1/conversations', { method: 'DELETE' });
        console.log('Delete response:', resp.status);
        currentConversationId = null;
        showHero();
        await loadConversations();
    } catch (e) {
        console.error('Delete error:', e);
    }
}

function startNewChat() {
    currentConversationId = null;
    conversationStarted = false;
    currentMessages = []; // تنظيف المصفوفة
    showHero();
    renderConversationsList();
}

newChatBtn.addEventListener('click', startNewChat);
clearAllChatsBtn.addEventListener('click', clearAllConversations);

function getCurrentInput() {
    return chatPanel.classList.contains('hidden') ? userInput : userInputChat;
}

function getCurrentSendBtn() {
    return chatPanel.classList.contains('hidden') ? sendBtn : sendBtnChat;
}

// ===== Attachments =====
function renderAttachments() {
    // Get the appropriate container based on current view
    const inHero = chatPanel.classList.contains('hidden');
    const holder = inHero ? document.getElementById('attachments') : document.getElementById('chatAttachments');
    
    if (!holder) return;
    
    holder.innerHTML = '';
    attachments.forEach((att, idx) => {
        const chip = document.createElement('div');
        const unreadable = !att.content || !att.content.trim();
        chip.className = `attach-chip${unreadable ? ' attach-chip--error' : ''}`;
        chip.innerHTML = `
            <span class="attach-name">${att.name}${unreadable ? ' (غير مقروء)' : ''}</span>
            <button aria-label="remove" data-idx="${idx}" class="attach-remove">×</button>
        `;
        holder.appendChild(chip);
    });
    holder.classList.toggle('hidden', attachments.length === 0);
    
    // Also update the other holder to keep in sync
    const otherHolder = inHero ? document.getElementById('chatAttachments') : document.getElementById('attachments');
    if (otherHolder) {
        otherHolder.innerHTML = holder.innerHTML;
        otherHolder.classList.toggle('hidden', attachments.length === 0);
    }
}

async function loadHistory() {
    // History is now managed through conversations
}

// ===== Send Message =====
async function sendMessage() {
    if (isGenerating) return;
    
    const activeInput = getCurrentInput();
    const activeBtn = getCurrentSendBtn();
    const text = activeInput.value.trim();
    const hasUnreadable = attachments.some(a => !a.content || !a.content.trim());
    if (hasUnreadable) {
        alert(translations[currentLang].fileReadError);
        return;
    }

    const contextFromFiles = attachments.map(a => `ملف: ${a.name}\n${a.content}`).join('\n\n');
    const combined = contextFromFiles ? `${text}\n\n${contextFromFiles}` : text;
    if (!combined) return;

    isGenerating = true;
    userScrolledUp = false; // إعادة تعيين عند بدء رسالة جديدة
    activeInput.value = '';
    activeInput.disabled = true;
    userInput.disabled = true;
    userInputChat.disabled = true;
    
    // Switch to chat view on first message
    if (chatPanel.classList.contains('hidden')) {
        revealChat();
    }
    
    // Show user message with file names only (not content)
    let displayText = text;
    if (attachments.length > 0) {
        const fileNames = attachments.map(a => `<span class="file-tag"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>${a.name}</span>`).join(' ');
        displayText = text ? `${text}<div class="file-tags">${fileNames}</div>` : `<div class="file-tags">${fileNames}</div>`;
    }
    addMessage(displayText, false, attachments.length > 0);
    
    // Clear attachments after sending
    attachments = [];
    renderAttachments();
    sendBtn.disabled = true;
    sendBtnChat.disabled = true;

    const aiDiv = addMessage('...', true);
    let fullText = '';

    // إضافة الرسالة الجديدة للمصفوفة
    currentMessages.push({ role: 'user', content: combined });
    
    // نسخ المصفوفة لإرسالها
    const allMessages = [...currentMessages];

    try {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: allMessages,
                conversation_id: currentConversationId,
                stream: true
            })
        });

        const contentType = response.headers.get('content-type') || '';
        aiDiv.innerText = '';

        if (!response.ok) {
            let errMsg = translations[currentLang].error;
            try {
                const errData = await response.json();
                if (errData?.error) errMsg = `❌ ${errData.error}`;
            } catch (e) {}
            fullText = errMsg;
            renderContent(fullText, aiDiv);
            return;
        }

        if (!response.body || contentType.includes('application/json')) {
            const data = await response.json();
            const msg = data?.choices?.[0]?.message?.content || (data?.error ? `❌ ${data.error}` : '') || '';
            if (data?.conversation_id) {
                currentConversationId = data.conversation_id;
            }
            fullText = msg || translations[currentLang].error;
            renderContent(fullText, aiDiv);
        } else {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        try {
                            const data = JSON.parse(line.substring(6));
                            if (data.conversation_id) {
                                currentConversationId = data.conversation_id;
                                // تحديث قائمة المحادثات فوراً
                                loadConversations();
                            } else if (data.choices && data.choices[0].delta.content) {
                                const content = data.choices[0].delta.content || '';
                                fullText += content;
                                renderContent(fullText, aiDiv);
                            }
                        } catch (e) {}
                    }
                }
                scrollToBottom();
            }
        }
    } catch (err) {
        renderContent(translations[currentLang].error, aiDiv);
        fullText = translations[currentLang].error;
    } finally {
        // تخزين رد المساعد في المصفوفة
        if (fullText) {
            currentMessages.push({ role: 'assistant', content: fullText });
        }
        isGenerating = false;
        sendBtn.disabled = false;
        sendBtnChat.disabled = false;
        userInput.disabled = false;
        userInputChat.disabled = false;
        attachments = [];
        renderAttachments();
        userInputChat.focus();
        await loadConversations();
    }
}

sendBtn.onclick = sendMessage;
sendBtnChat.onclick = sendMessage;

// دالة لتعديل ارتفاع textarea تلقائياً
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// ===== Event Listeners =====
// Shift+Enter للسطر الجديد، Enter للإرسال
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
userInput.addEventListener('input', () => autoResize(userInput));

userInputChat.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
userInputChat.addEventListener('input', () => autoResize(userInputChat));

// ===== Language =====
function applyLanguage(lang) {
    const t = translations[lang];
    currentLang = lang;
    document.documentElement.lang = t.lang;
    document.documentElement.dir = t.dir;
    document.body.classList.toggle('dir-ltr', lang === 'en');
    document.body.classList.toggle('dir-rtl', lang === 'ar');
    titleText.textContent = t.title;
    subtitleText.textContent = t.subtitle;
    statusText.textContent = t.status;
    userInput.placeholder = t.placeholder;
    userInputChat.placeholder = t.placeholder;
    sendBtn.setAttribute('title', t.send);
    sendBtn.setAttribute('aria-label', t.send);
    sendBtnChat.setAttribute('title', t.send);
    sendBtnChat.setAttribute('aria-label', t.send);
    langToggle.textContent = t.toggleLabel;
    newChatText.textContent = t.newChat;
    chatsLabel.textContent = t.prevChats;
    
    // Hero title
    const heroTitle = document.getElementById('heroTitle');
    if (heroTitle) heroTitle.textContent = t.heroTitle;
    
    // Sidebar buttons
    clearAllChatsBtn.setAttribute('title', t.deleteAll);
    toggleSidebarBtn.setAttribute('title', t.hideSidebar);
    openSidebarBtn.setAttribute('title', t.openSidebar);
    openSidebarBtn.setAttribute('aria-label', t.openSidebar);
    
    renderConversationsList();
}

langToggle.addEventListener('click', () => {
    const nextLang = currentLang === 'ar' ? 'en' : 'ar';
    applyLanguage(nextLang);
    userInput.focus();
});

// ===== File Upload =====
uploadBtn.addEventListener('click', () => fileInput.click());
uploadBtnChat.addEventListener('click', () => fileInput.click());

// Drag and drop functionality
async function handleFileDrop(files) {
    if (!files.length) return;
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    try {
        const res = await fetch('/v1/upload', { method: 'POST', body: form });
        const data = await res.json();
        const received = data.files || [];
        attachments = attachments.concat(received);
        renderAttachments();
    } catch (err) {
        // ignore upload errors
    }
}

// Hero input drag and drop
const heroInput = document.getElementById('heroInputInitial');
const chatInputBar = document.querySelector('.chat-input-bar');

[heroInput, chatInputBar].forEach(element => {
    if (!element) return;
    
    element.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.classList.add('drag-over');
    });
    
    element.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.classList.remove('drag-over');
    });
    
    element.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        element.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files || []);
        await handleFileDrop(files);
    });
});

fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    await handleFileDrop(files);
    fileInput.value = '';
});

document.addEventListener('click', (e) => {
    const btn = e.target.closest('.attach-remove');
    if (!btn) return;
    const idx = Number(btn.dataset.idx);
    attachments.splice(idx, 1);
    renderAttachments();
});

// ===== Initialize =====
applyLanguage('ar');
userInput.focus();
loadHistory();
loadConversations();
