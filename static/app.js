/**
 * RAG Chatbot — Frontend Application
 * ====================================
 * Handles document upload, chat interaction, and UI state management.
 */

// ── State ────────────────────────────────────────────────────────────
const state = {
    sessionId: crypto.randomUUID(),
    isLoading: false,
    uploadedDocs: [],
};

// ── DOM Elements ─────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elements = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    documentsList: $('#documentsList'),
    emptyDocs: $('#emptyDocs'),
    docCount: $('#docCount'),
    messagesContainer: $('#messagesContainer'),
    welcomeScreen: $('#welcomeScreen'),
    messageInput: $('#messageInput'),
    sendBtn: $('#sendBtn'),
    clearHistoryBtn: $('#clearHistoryBtn'),
    clearAllBtn: $('#clearAllBtn'),
    statusBadge: $('#statusBadge'),
    toastContainer: $('#toastContainer'),
};

// ── Initialize ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initChat();
    initSidebar();
    fetchStats();
});

// ── Upload Handling ──────────────────────────────────────────────────
function initUpload() {
    const { uploadZone, fileInput } = elements;

    uploadZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
        fileInput.value = '';
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['pdf', 'txt', 'docx', 'csv'];

    if (!validExts.includes(ext)) {
        showToast(`Unsupported file type: .${ext}`, 'error');
        return;
    }

    // Show progress
    const progressEl = showUploadProgress(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();

        addDocumentToList(file.name, ext, data.chunks_created);
        updateDocCount(data.total_chunks_in_store);
        showToast(`"${file.name}" uploaded — ${data.chunks_created} chunks indexed`, 'success');

    } catch (error) {
        showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
        progressEl?.remove();
    }
}

function showUploadProgress(filename) {
    const el = document.createElement('div');
    el.className = 'upload-progress';
    el.innerHTML = `
        <div class="upload-progress-bar" style="width: 100%"></div>
        <div class="upload-progress-text">
            <div class="spinner"></div>
            Processing ${filename}...
        </div>
    `;
    elements.uploadZone.after(el);
    return el;
}

function addDocumentToList(name, ext, chunks) {
    elements.emptyDocs.style.display = 'none';

    const iconMap = { pdf: '📕', txt: '📝', docx: '📘', csv: '📊' };

    const docEl = document.createElement('div');
    docEl.className = 'doc-item';
    docEl.innerHTML = `
        <div class="doc-icon ${ext}">${iconMap[ext] || '📄'}</div>
        <div class="doc-info">
            <div class="doc-name" title="${name}">${name}</div>
            <div class="doc-meta">${chunks} chunks</div>
        </div>
    `;

    elements.documentsList.appendChild(docEl);
    state.uploadedDocs.push({ name, ext, chunks });
}

function updateDocCount(total) {
    elements.docCount.textContent = `${total} chunks`;
}

// ── Chat Handling ────────────────────────────────────────────────────
function initChat() {
    const { messageInput, sendBtn } = elements;

    messageInput.addEventListener('input', () => {
        // Auto-resize textarea
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';

        // Enable/disable send
        sendBtn.disabled = !messageInput.value.trim();
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !state.isLoading) {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // Action buttons
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
    elements.clearAllBtn.addEventListener('click', clearAll);
}

async function sendMessage() {
    const question = elements.messageInput.value.trim();
    if (!question || state.isLoading) return;

    // Hide welcome screen
    if (elements.welcomeScreen) {
        elements.welcomeScreen.remove();
    }

    // Add user message
    addMessage(question, 'user');

    // Clear input
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    elements.sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = showTypingIndicator();
    setStatus('Thinking...');
    state.isLoading = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                session_id: state.sessionId,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Chat request failed');
        }

        const data = await response.json();

        typingEl.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        typingEl.remove();
        addMessage(`⚠️ Error: ${error.message}`, 'assistant');
        showToast(error.message, 'error');
    } finally {
        state.isLoading = false;
        setStatus('Ready');
    }
}

function addMessage(content, role, sources = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatarEmoji = role === 'user' ? '👤' : '🤖';
    const formattedContent = role === 'assistant' ? formatMarkdown(content) : escapeHtml(content);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const chips = sources.map(s =>
            `<span class="source-chip">
                📄 ${s.filename}${s.page !== 'N/A' ? ` p.${s.page}` : ''}
                <span class="source-score">${(s.relevance_score * 100).toFixed(0)}%</span>
            </span>`
        ).join('');

        sourcesHtml = `
            <div class="sources-section">
                <div class="sources-title">📌 Sources</div>
                ${chips}
            </div>
        `;
    }

    messageEl.innerHTML = `
        <div class="message-avatar">${avatarEmoji}</div>
        <div class="message-content">
            ${formattedContent}
            ${sourcesHtml}
        </div>
    `;

    elements.messagesContainer.appendChild(messageEl);
    scrollToBottom();
}

function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.id = 'typingIndicator';
    el.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    elements.messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

// ── Markdown Formatting ──────────────────────────────────────────────
function formatMarkdown(text) {
    // Basic markdown-to-HTML conversion
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // Unordered lists
    html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = `<p>${html}</p>`;

    // Clean empty paragraphs
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>(<h[2-4]>)/g, '$1');
    html = html.replace(/(<\/h[2-4]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');
    html = html.replace(/<p>(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)<\/p>/g, '$1');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Sidebar ──────────────────────────────────────────────────────────
function initSidebar() {
    elements.sidebarToggle.addEventListener('click', toggleSidebar);
}

function toggleSidebar() {
    const sidebar = elements.sidebar;
    const isOpen = sidebar.classList.toggle('open');

    let overlay = document.querySelector('.sidebar-overlay');
    if (isOpen && !overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay active';
        overlay.addEventListener('click', toggleSidebar);
        document.body.appendChild(overlay);
    } else if (!isOpen && overlay) {
        overlay.remove();
    }
}

// ── API Helpers ──────────────────────────────────────────────────────
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (response.ok) {
            const data = await response.json();
            updateDocCount(data.total_chunks);
        }
    } catch (e) {
        // Server may not be running
    }
}

async function clearHistory() {
    try {
        await fetch('/api/clear-history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        // Clear chat UI
        elements.messagesContainer.innerHTML = '';
        elements.messagesContainer.appendChild(createWelcomeScreen());
        state.sessionId = crypto.randomUUID();
        showToast('Chat history cleared', 'info');
    } catch (e) {
        showToast('Failed to clear history', 'error');
    }
}

async function clearAll() {
    if (!confirm('This will remove all uploaded documents and chat history. Continue?')) return;

    try {
        await fetch('/api/clear', { method: 'POST' });

        // Reset UI
        elements.messagesContainer.innerHTML = '';
        elements.messagesContainer.appendChild(createWelcomeScreen());
        elements.documentsList.innerHTML = '<div class="empty-state" id="emptyDocs"><p>No documents uploaded yet</p></div>';
        updateDocCount(0);
        state.uploadedDocs = [];
        state.sessionId = crypto.randomUUID();
        showToast('All data cleared', 'info');
    } catch (e) {
        showToast('Failed to clear data', 'error');
    }
}

function createWelcomeScreen() {
    const div = document.createElement('div');
    div.className = 'welcome-screen';
    div.id = 'welcomeScreen';
    div.innerHTML = `
        <div class="welcome-icon">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="url(#welcomeGrad2)" stroke-width="1.2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                <circle cx="9" cy="10" r="1" fill="url(#welcomeGrad2)"/>
                <circle cx="12" cy="10" r="1" fill="url(#welcomeGrad2)"/>
                <circle cx="15" cy="10" r="1" fill="url(#welcomeGrad2)"/>
                <defs>
                    <linearGradient id="welcomeGrad2" x1="3" y1="3" x2="21" y2="21">
                        <stop stop-color="#818cf8"/>
                        <stop offset="1" stop-color="#c084fc"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <h2 class="welcome-title">Welcome to RAG Chat</h2>
        <p class="welcome-text">Upload your documents and ask questions. I'll answer using only the information from your files — no hallucinations, just facts.</p>
        <div class="welcome-features">
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h4>Multi-Format</h4>
                <p>PDF, TXT, DOCX, CSV</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h4>Grounded</h4>
                <p>Answers from your docs only</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📌</div>
                <h4>Cited Sources</h4>
                <p>Every claim is traceable</p>
            </div>
        </div>
    `;
    return div;
}

// ── Utilities ────────────────────────────────────────────────────────
function scrollToBottom() {
    const container = elements.messagesContainer;
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

function setStatus(text) {
    const badge = elements.statusBadge;
    badge.innerHTML = `<span class="status-dot"></span> ${text}`;
}

function showToast(message, type = 'info') {
    const iconMap = { success: '✅', error: '❌', info: 'ℹ️' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${iconMap[type]}</span>
        <span>${message}</span>
    `;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
