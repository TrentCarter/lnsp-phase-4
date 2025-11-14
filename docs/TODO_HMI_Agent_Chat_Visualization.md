# TODO: HMI Visualization for Agent Chat (Phase 3+)

**Status**: 🔴 Not Started
**Priority**: HIGH (Critical for Phase 3 visibility)
**Estimated Effort**: 3-4 hours
**Dependencies**: Phase 3 LLM integration (✅ Complete)

---

## Overview

The agent chat system (Phase 2-3) is fully functional with LLM-powered Q&A, but conversations are not visible in the HMI. Users cannot see:
- When agents are asking questions
- What questions are being asked
- How Architect is responding
- Thread lifecycle (active/completed/failed)

This creates a "black box" experience where intelligent conversations happen behind the scenes.

---

## What Needs to Be Done

### 1. Event Broadcasting (Backend)

**File**: `services/common/agent_chat.py`

Add event broadcasting for agent chat operations (similar to job_card events):

```python
# In AgentChatClient methods, add:
def send_message(self, thread_id, from_agent, to_agent, message_type, content, metadata=None):
    # ... existing logic ...

    # NEW: Broadcast event to HMI
    _emit_agent_chat_event('message_sent', {
        'thread_id': thread_id,
        'from_agent': from_agent,
        'to_agent': to_agent,
        'message_type': message_type,
        'content': content[:100],  # Truncate for event stream
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    })

def create_thread(self, run_id, parent_agent, child_agent, metadata=None):
    # ... existing logic ...

    # NEW: Broadcast thread creation
    _emit_agent_chat_event('thread_created', {
        'thread_id': thread.thread_id,
        'run_id': run_id,
        'parent_agent': parent_agent,
        'child_agent': child_agent,
        'metadata': metadata
    })

def close_thread(self, thread_id, status, result, error=None):
    # ... existing logic ...

    # NEW: Broadcast thread closure
    _emit_agent_chat_event('thread_closed', {
        'thread_id': thread_id,
        'status': status,
        'result': result,
        'error': error
    })
```

**Helper function** (add to `agent_chat.py`):

```python
def _emit_agent_chat_event(event_type: str, data: Dict[str, Any]) -> None:
    """Emit agent chat event to Event Stream for HMI visualization"""
    try:
        payload = {
            "event_type": f"agent_chat_{event_type}",
            "data": data
        }

        requests.post(
            "http://localhost:6102/broadcast",
            json=payload,
            timeout=1.0
        )
    except Exception as e:
        # Don't fail operations if event broadcast fails
        pass
```

### 2. Sequencer Timeline Updates (Frontend)

**File**: `services/webui/templates/sequencer.html`

Add new row types for agent chat events:

```javascript
// In renderRows() function, add new row types:

// Thread creation row
if (event.event_type === 'agent_chat_thread_created') {
    rows.push({
        time: event.timestamp,
        agent: event.data.parent_agent,
        type: 'thread_start',
        content: `Started conversation with ${event.data.child_agent}`,
        color: '#3b82f6',  // Blue for thread start
        icon: '💬',
        metadata: event.data.metadata
    });
}

// Question row
if (event.event_type === 'agent_chat_message_sent' && event.data.message_type === 'question') {
    rows.push({
        time: event.timestamp,
        agent: event.data.from_agent,
        type: 'question',
        content: `❓ ${event.data.content}`,
        color: '#f59e0b',  // Amber for questions
        icon: '❓',
        urgency: event.data.metadata?.urgency,
        to_agent: event.data.to_agent
    });
}

// Answer row
if (event.event_type === 'agent_chat_message_sent' && event.data.message_type === 'answer') {
    rows.push({
        time: event.timestamp,
        agent: event.data.from_agent,
        type: 'answer',
        content: `💡 ${event.data.content}`,
        color: '#10b981',  // Green for answers
        icon: '💡',
        to_agent: event.data.to_agent
    });
}

// Status update row
if (event.event_type === 'agent_chat_message_sent' && event.data.message_type === 'status') {
    rows.push({
        time: event.timestamp,
        agent: event.data.from_agent,
        type: 'status',
        content: `${event.data.content}`,
        color: '#6b7280',  // Gray for status
        progress: event.data.metadata?.progress
    });
}

// Thread completion row
if (event.event_type === 'agent_chat_thread_closed') {
    const statusColor = event.data.status === 'completed' ? '#10b981' : '#ef4444';
    rows.push({
        time: event.timestamp,
        agent: 'System',
        type: 'thread_end',
        content: `Thread ${event.data.status}: ${event.data.result}`,
        color: statusColor,
        icon: event.data.status === 'completed' ? '✅' : '❌'
    });
}
```

**Add visual connectors** between question/answer pairs:

```javascript
// In renderSequencerRow(), add:
function renderSequencerRow(row) {
    const div = document.createElement('div');
    div.className = 'sequencer-row';

    // Add connector line for Q&A pairs
    if (row.type === 'answer' && previousRow?.type === 'question') {
        div.classList.add('answer-row');
        div.style.borderLeft = '2px solid #10b981';
        div.style.marginLeft = '20px';
    }

    // ... rest of rendering ...
}
```

### 3. TRON Tree View Animations (Frontend)

**File**: `services/webui/templates/tron.html`

Add message flow animations between parent and child nodes:

```javascript
// In updateTRON() function, add:

// Listen for agent chat events
eventSource.addEventListener('agent_chat_message_sent', (e) => {
    const data = JSON.parse(e.data);

    // Find parent and child nodes
    const fromNode = findNodeByAgent(data.from_agent);
    const toNode = findNodeByAgent(data.to_agent);

    if (fromNode && toNode) {
        // Animate message flow
        animateMessageFlow(fromNode, toNode, data.message_type);

        // Update node badges
        updateNodeBadge(toNode, data.message_type);
    }
});

function animateMessageFlow(fromNode, toNode, messageType) {
    const svg = document.getElementById('tron-svg');

    // Create animated dot along edge
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('r', 4);
    dot.setAttribute('class', `message-dot message-${messageType}`);

    // Calculate path from fromNode to toNode
    const fromX = fromNode.x;
    const fromY = fromNode.y;
    const toX = toNode.x;
    const toY = toNode.y;

    // Animate along path (1 second duration)
    const animation = dot.animate([
        { cx: fromX, cy: fromY },
        { cx: toX, cy: toY }
    ], {
        duration: 1000,
        easing: 'ease-in-out'
    });

    svg.appendChild(dot);

    // Remove dot after animation
    animation.onfinish = () => dot.remove();
}

function updateNodeBadge(node, messageType) {
    // Add temporary badge showing message type
    const badge = document.createElement('div');
    badge.className = `node-badge badge-${messageType}`;
    badge.textContent = messageType === 'question' ? '❓' :
                        messageType === 'answer' ? '💡' :
                        messageType === 'status' ? '📊' : '📝';

    node.element.appendChild(badge);

    // Fade out after 3 seconds
    setTimeout(() => {
        badge.style.opacity = '0';
        setTimeout(() => badge.remove(), 500);
    }, 3000);
}
```

**Add CSS for message animations**:

```css
.message-dot {
    fill: #3b82f6;
    opacity: 0.8;
}

.message-dot.message-question {
    fill: #f59e0b;
}

.message-dot.message-answer {
    fill: #10b981;
}

.message-dot.message-status {
    fill: #6b7280;
}

.node-badge {
    position: absolute;
    top: -10px;
    right: -10px;
    font-size: 16px;
    background: white;
    border-radius: 50%;
    padding: 2px;
    transition: opacity 0.5s;
}
```

### 4. Thread Detail Panel (Optional Enhancement)

**File**: `services/webui/templates/sequencer.html`

Add a sidebar panel showing active thread details when clicked:

```html
<div id="thread-detail-panel" class="hidden">
    <h3>Conversation Thread</h3>
    <div id="thread-messages">
        <!-- Populated dynamically -->
    </div>
    <div id="thread-metadata">
        <!-- Thread status, budget, etc. -->
    </div>
</div>
```

```javascript
function showThreadDetails(threadId) {
    // Fetch thread from backend
    fetch(`/api/agent_chat/thread/${threadId}`)
        .then(r => r.json())
        .then(thread => {
            renderThreadMessages(thread.messages);
            renderThreadMetadata(thread.metadata);
            document.getElementById('thread-detail-panel').classList.remove('hidden');
        });
}
```

---

## Event Types Summary

| Event Type | Data | Visual Representation |
|------------|------|----------------------|
| `agent_chat_thread_created` | thread_id, parent, child, metadata | Blue marker in Sequencer |
| `agent_chat_message_sent` (delegation) | from, to, content | Standard delegation row |
| `agent_chat_message_sent` (question) | from, to, content, urgency | Amber row with ❓ icon |
| `agent_chat_message_sent` (answer) | from, to, content | Green row with 💡 icon |
| `agent_chat_message_sent` (status) | from, to, content, progress | Gray row with progress % |
| `agent_chat_message_sent` (completion) | from, to, content, metadata | Success/failure indicator |
| `agent_chat_thread_closed` | thread_id, status, result | Green/red marker |

---

## Testing Checklist

After implementation, verify:

- [ ] Thread creation appears in Sequencer timeline
- [ ] Questions show with amber color and ❓ icon
- [ ] Answers show with green color and 💡 icon
- [ ] Status updates show progress percentage
- [ ] Thread completion shows success/failure badge
- [ ] TRON Tree View animates message flows between nodes
- [ ] Message dots travel along edges (parent→child→parent)
- [ ] Node badges appear temporarily on message receipt
- [ ] Thread detail panel (if implemented) shows full conversation
- [ ] No performance degradation with multiple active threads

---

## Example Visual Flow

**Sequencer Timeline:**
```
10:30:00 | Architect    | 💬 Started conversation with Dir-Code
10:30:01 | Architect    | → Refactor authentication to OAuth2
10:30:02 | Dir-Code     | ❓ Which files should I focus on? (urgency: blocking)
10:30:05 | Architect    | 💡 Focus on src/auth.py and src/oauth.py first
10:30:06 | Dir-Code     | 📊 Decomposing task... (30%)
10:30:10 | Dir-Code     | 📊 Delegating to Managers... (50%)
10:30:20 | Dir-Code     | ✅ Task completed successfully!
10:30:21 | System       | ✅ Thread completed: Successfully refactored to OAuth2
```

**TRON Tree View:**
```
     Architect
        ↓ 💬 (blue dot travels down)
     Dir-Code [❓ badge appears]
        ↑ 💡 (green dot travels up)
     Architect [💡 badge appears]
        ↓ 📊 (gray dot travels down)
     Dir-Code [📊 badge appears]
```

---

## Files to Modify

1. `services/common/agent_chat.py` - Add event broadcasting
2. `services/webui/templates/sequencer.html` - Add agent chat row types
3. `services/webui/templates/tron.html` - Add message flow animations
4. `services/webui/static/css/main.css` - Add styling for new elements

---

## Estimated Timeline

- **Event Broadcasting**: 1 hour
- **Sequencer Updates**: 1-2 hours
- **TRON Animations**: 1-2 hours
- **Testing & Polish**: 30 minutes

**Total**: 3.5-4.5 hours

---

## Priority Justification

This is **HIGH PRIORITY** because:

1. Phase 3 LLM integration is complete but invisible to users
2. Users cannot debug agent conversations without HMI visibility
3. Critical for demonstrating intelligent Q&A capabilities
4. Required before extending to other Directors (Phase 5)
5. Improves user trust by showing reasoning and decision-making

**Recommendation**: Implement before starting Phase 4 (real-time communication) to ensure complete visibility of current functionality.
