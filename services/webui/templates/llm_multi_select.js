// Multi-select history functionality for LLM interface
// Add these functions to the existing llm.html file

// Add to state management
let selectedSessions = new Set();

// Add these DOM element references
const selectAllBtn = document.getElementById('select-all-btn');
const deleteSelectedBtn = document.getElementById('delete-selected-btn');
const deleteAllBtn = document.getElementById('delete-all-btn');

// Add event listeners in setupEventListeners
function setupMultiSelectEvents() {
    selectAllBtn.addEventListener('click', selectAllSessions);
    deleteSelectedBtn.addEventListener('click', deleteSelectedSessions);
    deleteAllBtn.addEventListener('click', deleteAllSessions);
}

// Updated createSessionItem with checkboxes
function createSessionItemWithCheckbox(session) {
    const div = document.createElement('div');
    div.className = 'session-item';
    div.dataset.sessionId = session.session_id;
    if (session.session_id === currentSessionId) {
        div.classList.add('active');
    }
    if (session.status === 'archived') {
        div.classList.add('archived');
    }

    const messageCount = session.metadata?.message_count || 0;
    const timestamp = formatRelativeTime(session.updated_at);
    const title = session.title || 'New conversation';
    const titleClass = session.title ? '' : 'untitled';
    const archivedBadge = session.status === 'archived' ? '<span class="archived-badge">📦</span>' : '';

    div.innerHTML = `
        <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
            <input type="checkbox" class="session-checkbox" data-session-id="${session.session_id}" 
                   style="margin-top: 0.25rem; cursor: pointer; accent-color: #3b82f6;" 
                   onchange="handleCheckboxChange(this, '${session.session_id}')">
            <div style="flex: 1; min-width: 0;">
                <div class="session-item-header">
                    <div class="session-title ${titleClass}" title="${escapeHtml(title)}">
                        ${archivedBadge}${escapeHtml(title)}
                    </div>
                    <div class="session-actions">
                        <button class="session-action-btn" onclick="renameSession('${session.session_id}')" title="Rename">
                            ✎
                        </button>
                        <button class="session-action-btn" onclick="exportSession('${session.session_id}')" title="Export">
                            ↓
                        </button>
                        <button class="session-action-btn" onclick="deleteSession('${session.session_id}')" title="Delete">
                            ✕
                        </button>
                    </div>
                </div>
                <div class="session-meta">
                    <div class="session-agent">
                        <span>${getAgentIcon(session.agent_id)}</span>
                        <span>${session.agent_name}</span>
                    </div>
                    <div class="session-timestamp">${timestamp}</div>
                </div>
            </div>
        </div>
    `;

    // Handle checkbox sync
    const checkbox = div.querySelector('.session-checkbox');
    if (selectedSessions.has(session.session_id)) {
        checkbox.checked = true;
        div.classList.add('selected');
    }

    return div;
}

// Checkbox change handler
function handleCheckboxChange(checkbox, sessionId) {
    if (checkbox.checked) {
        selectedSessions.add(sessionId);
        checkbox.closest('.session-item').classList.add('selected');
    } else {
        selectedSessions.delete(sessionId);
        checkbox.closest('.session-item').classList.remove('selected');
    }
    updateDeleteSelectedButton();
    updateSelectAllButton();
}

// Multi-select functions
function selectAllSessions() {
    const sessionItems = document.querySelectorAll('.session-item');
    const allSelected = sessionItems.length === selectedSessions.size;
    
    if (allSelected) {
        // Deselect all
        selectedSessions.clear();
        sessionItems.forEach(item => {
            item.classList.remove('selected');
            const checkbox = item.querySelector('.session-checkbox');
            if (checkbox) checkbox.checked = false;
        });
        selectAllBtn.textContent = '📋 Select All';
    } else {
        // Select all
        selectedSessions.clear();
        sessionItems.forEach(item => {
            const sessionId = item.dataset.sessionId;
            if (sessionId) {
                selectedSessions.add(sessionId);
                item.classList.add('selected');
                const checkbox = item.querySelector('.session-checkbox');
                if (checkbox) checkbox.checked = true;
            }
        });
        selectAllBtn.textContent = '❌ Deselect All';
    }
    
    updateDeleteSelectedButton();
}

function updateDeleteSelectedButton() {
    deleteSelectedBtn.disabled = selectedSessions.size === 0;
    deleteSelectedBtn.textContent = `🗑️ Delete Selected (${selectedSessions.size})`;
}

function updateSelectAllButton() {
    const sessionItems = document.querySelectorAll('.session-item');
    const allSelected = sessionItems.length > 0 && sessionItems.length === selectedSessions.size;
    selectAllBtn.textContent = allSelected ? '❌ Deselect All' : '📋 Select All';
}

async function deleteSelectedSessions() {
    if (selectedSessions.size === 0) return;
    
    const confirmed = confirm(`Delete ${selectedSessions.size} selected conversation(s)? This cannot be undone.`);
    if (!confirmed) return;

    let deletedCount = 0;
    const promises = Array.from(selectedSessions).map(async sessionId => {
        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            const data = await response.json();
            if (data.status === 'deleted') {
                deletedCount++;
                return true;
            }
            return false;
        } catch (error) {
            console.error(`Error deleting session ${sessionId}:`, error);
            return false;
        }
    });

    await Promise.all(promises);
    
    // Clear selection and refresh
    selectedSessions.clear();
    updateDeleteSelectedButton();
    updateSelectAllButton();
    loadSessions();
    
    // If current session was deleted, reset
    if (currentSessionId && !Array.from(document.querySelectorAll('.session-item')).some(item => item.dataset.sessionId === currentSessionId)) {
        currentSessionId = null;
        clearMessages();
    }
}

async function deleteAllSessions() {
    const confirmed = confirm('⚠️ Delete ALL conversation history? This cannot be undone.');
    if (!confirmed) return;

    const doubleConfirmed = confirm('⚠️ FINAL WARNING: This will permanently delete ALL conversations. Are you absolutely sure?');
    if (!doubleConfirmed) return;

    try {
        const response = await fetch('/api/chat/sessions', {
            method: 'DELETE'
        });
        const data = await response.json();
        
        if (data.status === 'ok') {
            selectedSessions.clear();
            currentSessionId = null;
            clearMessages();
            loadSessions();
            alert('All conversations deleted successfully.');
        } else {
            alert('Failed to delete all conversations: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting all sessions:', error);
        alert('Failed to delete all conversations.');
    }
}

// CSS styles for multi-select
const multiSelectStyles = `
    .session-item.selected {
        background: rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }

    .session-checkbox {
        accent-color: #3b82f6;
        width: 16px;
        height: 16px;
        cursor: pointer;
    }

    .session-item:hover .session-checkbox {
        visibility: visible;
    }

    .session-item .session-checkbox {
        visibility: hidden;
    }

    .session-item.selected .session-checkbox,
    .session-item:hover .session-checkbox {
        visibility: visible;
    }
`;

// Inject styles
const style = document.createElement('style');
style.textContent = multiSelectStyles;
document.head.appendChild(style);
