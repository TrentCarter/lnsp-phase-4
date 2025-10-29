"""
Master Chat UI - Unified Interface for All LVM Models

Port 9000: Master chat interface (Claude.ai/ChatGPT style)
Routes to: Ports 9001-9006 (AMN, Transformer, GRU, LSTM, Vec2Text, Transformer Optimized)

Features:
- Modern chat interface with message bubbles
- Left sidebar with chat history and model selector
- Settings panel (steps=3 default, sentence chunking, conversation context)
- Dark mode toggle
- Export, copy, regenerate features
- Token/vector counter

Usage:
    uvicorn app.api.master_chat:app --host 127.0.0.1 --port 9000 --reload
"""

import uuid
import json
import httpx
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Master Chat UI", version="1.0.0")

# ============================================================================
# Configuration
# ============================================================================

MODEL_BACKENDS = {
    "AMN": "http://localhost:9001",
    "Transformer (Baseline)": "http://localhost:9002",
    "GRU": "http://localhost:9003",
    "LSTM ⭐": "http://localhost:9004",
    "Vec2Text Direct": "http://localhost:9005",
    "Transformer (Optimized)": "http://localhost:9006",
}

# In-memory chat history (replace with DB for persistence)
chat_history: Dict[str, List[Dict]] = {}
chat_metadata: Dict[str, Dict] = {}  # Stores chat titles, model, created_at

# ============================================================================
# Data Models
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    latency_ms: Optional[float] = None
    confidence: Optional[float] = None
    model: Optional[str] = None  # Model used for this message

class ChatRequest(BaseModel):
    """Request to send a message"""
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    model: str = "LSTM ⭐"
    use_conversation_context: bool = True
    chunk_mode: str = "sentence"  # sentence, adaptive, fixed, off
    decode_steps: int = 3
    temperature: float = 1.0

class ChatHistoryResponse(BaseModel):
    """Response containing chat history"""
    chat_id: str
    messages: List[ChatMessage]
    title: str
    model: str
    created_at: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to chat interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/chat" />
    </head>
    <body>
        <p>Redirecting to chat interface...</p>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check"""
    # Check all model backends
    backend_status = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for model_name, url in MODEL_BACKENDS.items():
            try:
                response = await client.get(f"{url}/health")
                backend_status[model_name] = "healthy" if response.status_code == 200 else "degraded"
            except:
                backend_status[model_name] = "unavailable"

    return {
        "status": "healthy",
        "backends": backend_status,
        "total_chats": len(chat_history)
    }

@app.get("/models")
async def list_models():
    """List available models and their status"""
    models = []
    async with httpx.AsyncClient(timeout=2.0) as client:
        for model_name, url in MODEL_BACKENDS.items():
            try:
                response = await client.get(f"{url}/health")
                status = "healthy" if response.status_code == 200 else "degraded"
            except:
                status = "unavailable"

            models.append({
                "name": model_name,
                "url": url,
                "status": status
            })

    return {"models": models}

@app.post("/chat/send")
async def send_message(request: ChatRequest):
    """Send a message and get response"""
    # Get or create chat history
    if request.chat_id not in chat_history:
        chat_history[request.chat_id] = []
        chat_metadata[request.chat_id] = {
            "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
            "model": request.model,
            "created_at": datetime.now().isoformat()
        }

    # Add user message to history
    user_msg = ChatMessage(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    chat_history[request.chat_id].append(user_msg.dict())

    # Prepare messages for backend
    if request.use_conversation_context:
        # Use last N user messages (max 5 for context window)
        user_messages = [msg["content"] for msg in chat_history[request.chat_id] if msg["role"] == "user"]
        # Take last 5 messages max (to fit within context length)
        context_messages = user_messages[-5:] if len(user_messages) > 5 else user_messages
    else:
        # Use only current message
        context_messages = [request.message]

    # Route to appropriate model backend
    backend_url = MODEL_BACKENDS.get(request.model)
    if not backend_url:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    # Call backend
    try:
        backend_payload = {
            "messages": context_messages,
            "decode_steps": request.decode_steps,
            "chunk_mode": request.chunk_mode,
            "temperature": request.temperature,
            "auto_chunk": True  # Enable auto-chunking for long messages
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{backend_url}/chat",
                json=backend_payload
            )

            # Log errors for debugging
            if response.status_code == 422:
                print(f"❌ Backend validation error:")
                print(f"   Backend: {backend_url}")
                print(f"   Payload: {backend_payload}")
                print(f"   Response: {response.text}")

            response.raise_for_status()
            result = response.json()

        # Add assistant message to history
        assistant_msg = ChatMessage(
            role="assistant",
            content=result["response"],
            timestamp=datetime.now().isoformat(),
            latency_ms=result.get("total_latency_ms"),
            confidence=result.get("confidence")
        )
        chat_history[request.chat_id].append(assistant_msg.dict())

        return {
            "chat_id": request.chat_id,
            "message": assistant_msg,
            "latency_ms": result.get("total_latency_ms"),
            "confidence": result.get("confidence"),
            "chunks_used": result.get("chunks_used")
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")

@app.get("/chat/history/{chat_id}")
async def get_chat_history(chat_id: str):
    """Get chat history for a specific chat"""
    if chat_id not in chat_history:
        raise HTTPException(status_code=404, detail="Chat not found")

    metadata = chat_metadata.get(chat_id, {})
    return ChatHistoryResponse(
        chat_id=chat_id,
        messages=[ChatMessage(**msg) for msg in chat_history[chat_id]],
        title=metadata.get("title", "Untitled Chat"),
        model=metadata.get("model", "Unknown"),
        created_at=metadata.get("created_at", "")
    )

@app.get("/chat/list")
async def list_chats():
    """List all chats"""
    chats = []
    for chat_id, metadata in chat_metadata.items():
        msg_count = len(chat_history.get(chat_id, []))
        chats.append({
            "chat_id": chat_id,
            "title": metadata.get("title", "Untitled"),
            "model": metadata.get("model", "Unknown"),
            "created_at": metadata.get("created_at", ""),
            "message_count": msg_count
        })

    # Sort by created_at (most recent first)
    chats.sort(key=lambda x: x["created_at"], reverse=True)
    return {"chats": chats}

@app.delete("/chat/delete/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    if chat_id in chat_history:
        del chat_history[chat_id]
    if chat_id in chat_metadata:
        del chat_metadata[chat_id]

    return {"status": "deleted", "chat_id": chat_id}

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Master chat interface (Claude.ai/ChatGPT style)"""
    # Return with cache-busting headers to ensure browser gets latest version
    return HTMLResponse(
        content=CHAT_UI_HTML,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


# ============================================================================
# Chat UI HTML (Modern Claude.ai / ChatGPT Style)
# ============================================================================

CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LVM Master Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f7f9fc;
            --bg-sidebar: #f0f2f5;
            --text-primary: #2d3748;
            --text-secondary: #718096;
            --border-color: #e2e8f0;
            --accent-color: #667eea;
            --accent-hover: #5a67d8;
            --user-msg-bg: #667eea;
            --assistant-msg-bg: #f7f9fc;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        [data-theme="dark"] {
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --bg-sidebar: #252f3f;
            --text-primary: #e2e8f0;
            --text-secondary: #a0aec0;
            --border-color: #4a5568;
            --user-msg-bg: #5a67d8;
            --assistant-msg-bg: #2d3748;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        /* Left Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .new-chat-btn {
            width: 100%;
            padding: 12px 16px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        .new-chat-btn:hover {
            background: var(--accent-hover);
        }

        .model-selector {
            margin-top: 12px;
            width: 100%;
            padding: 10px 12px;
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }

        .chat-list {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }

        .chat-item {
            padding: 12px;
            margin-bottom: 8px;
            background: var(--bg-primary);
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
            border: 1px solid transparent;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-item:hover {
            background: var(--bg-secondary);
            border-color: var(--border-color);
        }

        .chat-item:hover .chat-delete-btn {
            opacity: 1;
        }

        .chat-item.active {
            border-color: var(--accent-color);
        }

        .chat-item-content {
            flex: 1;
            min-width: 0;
        }

        .chat-item-title {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .chat-item-meta {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .chat-delete-btn {
            opacity: 0;
            background: transparent;
            border: none;
            color: #f56565;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 16px;
            transition: opacity 0.2s, background 0.2s;
            flex-shrink: 0;
            margin-left: 8px;
        }

        .chat-delete-btn:hover {
            background: rgba(245, 101, 101, 0.1);
        }

        .sidebar-footer {
            padding: 16px;
            border-top: 1px solid var(--border-color);
        }

        .theme-toggle {
            width: 100%;
            padding: 10px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            color: var(--text-primary);
        }

        /* Main Chat Area */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chat-meta {
            display: flex;
            gap: 12px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .model-badge {
            padding: 4px 10px;
            background: var(--accent-color);
            color: white;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }

        .stat-badge {
            padding: 4px 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 11px;
            font-weight: 500;
            color: var(--text-primary);
        }

        .stat-label {
            color: var(--text-secondary);
            font-weight: 400;
            margin-right: 4px;
        }

        .stat-value {
            font-weight: 600;
            color: var(--accent-color);
        }

        .confidence-high { color: #48bb78; }
        .confidence-medium { color: #ed8936; }
        .confidence-low { color: #f56565; }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .message {
            margin-bottom: 24px;
            display: flex;
            gap: 12px;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--accent-color);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: #48bb78;
        }

        .message-content {
            max-width: 65%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 14px;
        }

        .message.user .message-content {
            background: var(--user-msg-bg);
            color: white;
        }

        .message.assistant .message-content {
            background: var(--assistant-msg-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .message-meta {
            margin-top: 6px;
            font-size: 11px;
            color: var(--text-secondary);
            display: flex;
            gap: 12px;
        }

        .message-actions {
            margin-top: 8px;
            display: flex;
            gap: 8px;
        }

        .message-action-btn {
            padding: 4px 10px;
            background: none;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 11px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .message-action-btn:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        /* Input Area */
        .input-area {
            padding: 20px 24px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-primary);
        }

        .input-controls {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .control-group label {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .control-group input[type="checkbox"] {
            cursor: pointer;
        }

        .control-group select,
        .control-group input[type="number"] {
            padding: 6px 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 12px;
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
        }

        .input-box {
            flex: 1;
            padding: 14px 16px;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            min-height: 50px;
            max-height: 200px;
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        .send-btn {
            padding: 0 24px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-color);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .intro {
            text-align: center;
            padding: 60px 24px;
            color: var(--text-secondary);
        }

        .intro h2 {
            font-size: 28px;
            margin-bottom: 12px;
            color: var(--text-primary);
        }

        .intro p {
            font-size: 14px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <button class="new-chat-btn" onclick="newChat()">
                    ✨ New Chat
                </button>
                <select class="model-selector" id="modelSelector" onchange="updateModel()">
                    <option value="LSTM ⭐">LSTM ⭐ (Recommended)</option>
                    <option value="Transformer (Optimized)">Transformer (Optimized)</option>
                    <option value="Transformer (Baseline)">Transformer (Baseline)</option>
                    <option value="GRU">GRU</option>
                    <option value="AMN">AMN (Attention Mixer)</option>
                    <option value="Vec2Text Direct">Vec2Text Direct (No LVM)</option>
                </select>
            </div>
            <div class="chat-list" id="chatList">
                <!-- Chat history will be populated here -->
            </div>
            <div class="sidebar-footer">
                <button class="theme-toggle" onclick="toggleTheme()">
                    🌙 Toggle Dark Mode
                </button>
            </div>
        </div>

        <!-- Main Chat Area -->
        <div class="main">
            <div class="chat-header">
                <div class="chat-title" id="chatTitle">Welcome to LVM Master Chat</div>
                <div class="chat-meta">
                    <span class="model-badge" id="modelBadge">LSTM ⭐</span>
                    <span class="stat-badge" id="latencyDisplay"></span>
                    <span class="stat-badge" id="conceptsPerSecDisplay"></span>
                    <span class="stat-badge" id="tokensPerSecDisplay"></span>
                    <span class="stat-badge" id="confidenceDisplay"></span>
                </div>
            </div>

            <div class="messages" id="messages">
                <div class="intro">
                    <h2>👋 Hello! I'm your LVM assistant.</h2>
                    <p>I operate in 768D semantic space without tokens. Choose a model from the dropdown above and start chatting!</p>
                    <p style="margin-top: 12px; font-size: 12px;">💡 Try: "The Eiffel Tower was built in 1889" or "Explain photosynthesis"</p>
                </div>
            </div>

            <div class="input-area">
                <div class="input-controls">
                    <div class="control-group">
                        <input type="checkbox" id="useContext" checked>
                        <label for="useContext">Use conversation context</label>
                    </div>
                    <div class="control-group">
                        <label for="chunkMode">Chunking:</label>
                        <select id="chunkMode">
                            <option value="sentence" selected>By Sentence (1:1)</option>
                            <option value="adaptive">Adaptive</option>
                            <option value="fixed">Fixed (5 chunks)</option>
                            <option value="off">Off (Retrieval)</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="decodeSteps">Vec2Text Steps:</label>
                        <input type="number" id="decodeSteps" value="3" min="1" max="50" style="width: 60px;">
                    </div>
                </div>
                <div class="input-wrapper">
                    <textarea class="input-box" id="messageInput" placeholder="Type your message... (Enter to send, Shift+Enter for new line)" onkeypress="handleKeyPress(event)"></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentChatId = null;
        let currentModel = "LSTM ⭐";
        let isDarkMode = false;

        // Initialize
        async function init() {
            await loadChatList();
            newChat();
        }

        // New chat
        function newChat() {
            currentChatId = null;
            document.getElementById('messages').innerHTML = `
                <div class="intro">
                    <h2>👋 Hello! I'm your LVM assistant.</h2>
                    <p>I operate in 768D semantic space without tokens. Choose a model from the dropdown above and start chatting!</p>
                    <p style="margin-top: 12px; font-size: 12px;">💡 Try: "The Eiffel Tower was built in 1889" or "Explain photosynthesis"</p>
                </div>
            `;
            document.getElementById('chatTitle').textContent = 'New Chat';
            document.getElementById('messageInput').value = '';
            document.getElementById('messageInput').focus();
        }

        // Update model
        function updateModel() {
            currentModel = document.getElementById('modelSelector').value;
            document.getElementById('modelBadge').textContent = currentModel;
        }

        // Send message
        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();

            if (!message) return;

            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="loading"></span>';

            // Add user message to UI
            addMessage('user', message);
            messageInput.value = '';

            // Prepare request
            const useContext = document.getElementById('useContext').checked;
            const chunkMode = document.getElementById('chunkMode').value;
            const decodeSteps = parseInt(document.getElementById('decodeSteps').value);

            // Build request payload (omit chat_id if null to let backend generate UUID)
            const requestBody = {
                message: message,
                model: currentModel,
                use_conversation_context: useContext,
                chunk_mode: chunkMode,
                decode_steps: decodeSteps,
                temperature: 1.0
            };
            if (currentChatId) {
                requestBody.chat_id = currentChatId;
            }

            try {
                const response = await fetch('/chat/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const result = await response.json();

                // Update current chat ID and title
                if (!currentChatId) {
                    currentChatId = result.chat_id;
                    // Update title to first message (truncated)
                    const title = message.length > 50 ? message.substring(0, 50) + '...' : message;
                    document.getElementById('chatTitle').textContent = title;
                    await loadChatList();
                }

                // Add assistant message to UI
                addMessage('assistant', result.message.content, {
                    latency_ms: result.latency_ms,
                    confidence: result.confidence,
                    model: currentModel
                });

                // Update performance stats
                updateStats(result);

            } catch (error) {
                addMessage('assistant', `Error: ${error.message}`, { error: true });
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }

        // Add message to UI
        function addMessage(role, content, meta = {}) {
            const messagesDiv = document.getElementById('messages');

            // Remove intro if present
            const intro = messagesDiv.querySelector('.intro');
            if (intro) intro.remove();

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            // Use model name for assistant avatar, 'U' for user
            let avatar = 'U';
            if (role === 'assistant') {
                // Get short model name (first word or abbreviation)
                const modelName = meta.model || currentModel;
                if (modelName.includes('LSTM')) {
                    avatar = 'LSTM';
                } else if (modelName.includes('AMN')) {
                    avatar = 'AMN';
                } else if (modelName.includes('GRU')) {
                    avatar = 'GRU';
                } else if (modelName.includes('Transformer')) {
                    avatar = modelName.includes('Optimized') ? 'T-O' : 'T-B';
                } else if (modelName.includes('Vec2Text')) {
                    avatar = 'V2T';
                } else {
                    avatar = modelName.substring(0, 3).toUpperCase();
                }
            }
            const avatarColor = role === 'user' ? '#48bb78' : '#667eea';

            let metaHtml = '';
            if (meta.latency_ms || meta.confidence) {
                const parts = [];
                if (meta.latency_ms) parts.push(`${Math.round(meta.latency_ms)}ms`);
                if (meta.confidence) parts.push(`${(meta.confidence * 100).toFixed(1)}% confidence`);
                metaHtml = `<div class="message-meta">${parts.join(' • ')}</div>`;
            }

            let actionsHtml = '';
            if (role === 'assistant') {
                actionsHtml = `
                    <div class="message-actions">
                        <button class="message-action-btn" onclick="copyMessage(this)">📋 Copy</button>
                        <button class="message-action-btn" onclick="regenerateMessage()">🔄 Regenerate</button>
                    </div>
                `;
            }

            messageDiv.innerHTML = `
                <div class="message-avatar" style="background: ${avatarColor}">${avatar}</div>
                <div>
                    <div class="message-content">${escapeHtml(content)}</div>
                    ${metaHtml}
                    ${actionsHtml}
                </div>
            `;

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // Copy message
        function copyMessage(btn) {
            const content = btn.closest('.message').querySelector('.message-content').textContent;
            navigator.clipboard.writeText(content);
            btn.textContent = '✓ Copied';
            setTimeout(() => btn.textContent = '📋 Copy', 2000);
        }

        // Regenerate message
        async function regenerateMessage() {
            // Get last user message
            const messages = document.querySelectorAll('.message.user');
            if (messages.length === 0) return;

            const lastMessage = messages[messages.length - 1].querySelector('.message-content').textContent;

            // Remove last assistant message
            const assistantMessages = document.querySelectorAll('.message.assistant');
            if (assistantMessages.length > 0) {
                assistantMessages[assistantMessages.length - 1].remove();
            }

            // Resend
            document.getElementById('messageInput').value = lastMessage;
            await sendMessage();
        }

        // Load chat list
        async function loadChatList() {
            try {
                const response = await fetch('/chat/list');
                const data = await response.json();

                const chatList = document.getElementById('chatList');
                chatList.innerHTML = '';

                data.chats.forEach(chat => {
                    const chatItem = document.createElement('div');
                    chatItem.className = 'chat-item';
                    if (chat.chat_id === currentChatId) chatItem.classList.add('active');

                    const date = new Date(chat.created_at).toLocaleDateString();
                    chatItem.innerHTML = `
                        <div class="chat-item-content">
                            <div class="chat-item-title">${chat.title}</div>
                            <div class="chat-item-meta">${chat.model} • ${date} • ${chat.message_count} msgs</div>
                        </div>
                        <button class="chat-delete-btn" onclick="deleteChat('${chat.chat_id}', event)" title="Delete chat">🗑️</button>
                    `;

                    chatItem.onclick = () => loadChat(chat.chat_id);
                    chatList.appendChild(chatItem);
                });

            } catch (error) {
                console.error('Failed to load chat list:', error);
            }
        }

        // Load specific chat
        async function loadChat(chatId) {
            try {
                const response = await fetch(`/chat/history/${chatId}`);
                const data = await response.json();

                currentChatId = chatId;
                document.getElementById('chatTitle').textContent = data.title;
                document.getElementById('modelSelector').value = data.model;
                updateModel();

                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = '';

                data.messages.forEach(msg => {
                    addMessage(msg.role, msg.content, {
                        latency_ms: msg.latency_ms,
                        confidence: msg.confidence
                    });
                });

                await loadChatList();

            } catch (error) {
                console.error('Failed to load chat:', error);
            }
        }

        // Update performance stats display
        function updateStats(result) {
            // Latency
            if (result.latency_ms) {
                const latencyMs = Math.round(result.latency_ms);
                document.getElementById('latencyDisplay').innerHTML =
                    `<span class="stat-label">Latency:</span><span class="stat-value">${latencyMs}ms</span>`;
            }

            // Concepts/s (chunks used / time in seconds)
            if (result.chunks_used && result.latency_ms) {
                const conceptsPerSec = (result.chunks_used / (result.latency_ms / 1000)).toFixed(1);
                document.getElementById('conceptsPerSecDisplay').innerHTML =
                    `<span class="stat-label">Concepts/s:</span><span class="stat-value">${conceptsPerSec}</span>`;
            }

            // Equivalent tokens/s (assuming ~4 tokens per concept)
            if (result.chunks_used && result.latency_ms) {
                const tokensPerSec = ((result.chunks_used * 4) / (result.latency_ms / 1000)).toFixed(1);
                document.getElementById('tokensPerSecDisplay').innerHTML =
                    `<span class="stat-label">~Tokens/s:</span><span class="stat-value">${tokensPerSec}</span>`;
            }

            // Confidence (color-coded)
            if (result.confidence !== undefined) {
                const confidencePct = (result.confidence * 100).toFixed(1);
                let confidenceClass = 'confidence-medium';
                if (result.confidence >= 0.7) confidenceClass = 'confidence-high';
                if (result.confidence < 0.5) confidenceClass = 'confidence-low';

                document.getElementById('confidenceDisplay').innerHTML =
                    `<span class="stat-label">Confidence:</span><span class="stat-value ${confidenceClass}">${confidencePct}%</span>`;
            }
        }

        // Delete chat
        async function deleteChat(chatId, event) {
            event.stopPropagation(); // Prevent loading the chat when clicking delete

            if (!confirm('Delete this chat? This cannot be undone.')) {
                return;
            }

            try {
                const response = await fetch(`/chat/delete/${chatId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    // If we deleted the current chat, start a new one
                    if (chatId === currentChatId) {
                        newChat();
                    }
                    // Reload chat list
                    await loadChatList();
                } else {
                    alert('Failed to delete chat');
                }
            } catch (error) {
                console.error('Failed to delete chat:', error);
                alert('Error deleting chat');
            }
        }

        // Toggle dark mode
        function toggleTheme() {
            isDarkMode = !isDarkMode;
            document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
        }

        // Handle keyboard shortcuts
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize on load
        init();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
