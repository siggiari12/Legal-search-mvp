/**
 * Legal Search MVP - Frontend Application
 *
 * Handles chat interface interactions and API communication.
 */

// Configuration
const API_BASE_URL = window.location.origin;
const API_CHAT_ENDPOINT = `${API_BASE_URL}/api/chat`;

// DOM Elements
const chatForm = document.getElementById('chatForm');
const queryInput = document.getElementById('queryInput');
const submitButton = document.getElementById('submitButton');
const chatMessages = document.getElementById('chatMessages');

// State
let isLoading = false;

/**
 * Initialize the application
 */
function init() {
    // Form submission
    chatForm.addEventListener('submit', handleSubmit);

    // Keyboard shortcuts
    queryInput.addEventListener('keydown', handleKeydown);

    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);

    // Focus input on load
    queryInput.focus();
}

/**
 * Handle form submission
 */
async function handleSubmit(event) {
    event.preventDefault();

    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Add user message
    addMessage(query, 'user');

    // Clear input
    queryInput.value = '';
    autoResizeTextarea();

    // Show loading
    setLoading(true);
    const loadingMessage = addLoadingMessage();

    try {
        const response = await sendQuery(query);
        removeMessage(loadingMessage);
        handleResponse(response);
    } catch (error) {
        removeMessage(loadingMessage);
        handleError(error);
    } finally {
        setLoading(false);
        queryInput.focus();
    }
}

/**
 * Handle keyboard shortcuts
 */
function handleKeydown(event) {
    // Enter to submit (without Shift)
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
}

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
}

/**
 * Send query to API
 */
async function sendQuery(query) {
    const response = await fetch(API_CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
    }

    return response.json();
}

/**
 * Handle successful API response
 */
function handleResponse(data) {
    if (data.success && data.answer) {
        addAssistantMessage(data);
    } else if (data.clarification_question) {
        addClarificationMessage(data.clarification_question);
    } else if (data.failure_type) {
        addFailureMessage(data);
    } else {
        addFailureMessage({
            failure_type: 'unknown_error',
            message: 'Óvænt villa kom upp.',
        });
    }
}

/**
 * Handle API error
 */
function handleError(error) {
    console.error('API Error:', error);
    addFailureMessage({
        failure_type: 'connection_error',
        message: 'Ekki tókst að tengjast þjónustu. Vinsamlegast reyndu aftur.',
    });
}

/**
 * Add a message to the chat
 */
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<p>${escapeHtml(content)}</p>`;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

/**
 * Add assistant message with citations
 */
function addAssistantMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Answer text (convert markdown-like formatting)
    const answerHtml = formatAnswer(data.answer);
    contentDiv.innerHTML = answerHtml;

    // Confidence badge
    if (data.confidence) {
        const badge = document.createElement('div');
        badge.className = `confidence-badge confidence-${data.confidence}`;
        badge.textContent = getConfidenceLabel(data.confidence);
        contentDiv.appendChild(badge);
    }

    // Citations
    if (data.citations && data.citations.length > 0) {
        const citationsContainer = createCitationsContainer(data.citations);
        contentDiv.appendChild(citationsContainer);
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add clarification request message
 */
function addClarificationMessage(question) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="clarification-message">
            <p class="clarification-text">${escapeHtml(question)}</p>
        </div>
    `;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add failure message
 */
function addFailureMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const title = getFailureTitle(data.failure_type);
    const message = data.message || 'Óvænt villa kom upp.';

    contentDiv.innerHTML = `
        <div class="failure-message">
            <p class="failure-title">${escapeHtml(title)}</p>
            <p class="failure-text">${escapeHtml(message)}</p>
        </div>
    `;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add loading message
 */
function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content loading-message';
    contentDiv.innerHTML = `
        <span>Leita í lögum</span>
        <div class="loading-dots">
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
        </div>
    `;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

/**
 * Remove a message from chat
 */
function removeMessage(messageElement) {
    if (messageElement && messageElement.parentNode) {
        messageElement.parentNode.removeChild(messageElement);
    }
}

/**
 * Create citations container
 */
function createCitationsContainer(citations) {
    const container = document.createElement('div');
    container.className = 'citations-container';

    const header = document.createElement('div');
    header.className = 'citations-header';
    header.innerHTML = `
        <span class="citations-toggle">▶</span>
        <span>Heimildir (${citations.length})</span>
    `;

    const list = document.createElement('div');
    list.className = 'citations-list';

    citations.forEach((citation, index) => {
        const item = document.createElement('div');
        item.className = 'citation-item';
        item.innerHTML = `
            <div class="citation-locator">[${index + 1}] ${escapeHtml(citation.locator)}</div>
            ${citation.quote ? `<div class="citation-quote">"${escapeHtml(citation.quote)}"</div>` : ''}
        `;
        list.appendChild(item);
    });

    // Toggle functionality
    header.addEventListener('click', () => {
        const toggle = header.querySelector('.citations-toggle');
        toggle.classList.toggle('expanded');
        list.classList.toggle('expanded');
    });

    container.appendChild(header);
    container.appendChild(list);

    return container;
}

/**
 * Format answer text (basic markdown support)
 */
function formatAnswer(text) {
    if (!text) return '';

    // Escape HTML first
    let html = escapeHtml(text);

    // Convert line breaks to paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');

    // Convert single line breaks within paragraphs
    html = html.replace(/\n/g, '<br>');

    // Bold text
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Citation references [1], [2], etc.
    html = html.replace(/\[(\d+)\]/g, '<sup class="citation-ref">[$1]</sup>');

    return html;
}

/**
 * Get confidence label in Icelandic
 */
function getConfidenceLabel(confidence) {
    const labels = {
        high: 'Hátt öryggi',
        medium: 'Miðlungs öryggi',
        low: 'Lágt öryggi',
        none: 'Ekkert öryggi',
    };
    return labels[confidence] || confidence;
}

/**
 * Get failure title in Icelandic
 */
function getFailureTitle(failureType) {
    const titles = {
        ambiguous_query: 'Spurningin er of almenn',
        no_relevant_data: 'Engar heimildir fundust',
        validation_failed: 'Ekki tókst að staðfesta svar',
        rate_limited: 'Of margar fyrirspurnir',
        internal_error: 'Kerfisvilla',
        connection_error: 'Tengivilla',
        service_unavailable: 'Þjónusta ekki tiltæk',
    };
    return titles[failureType] || 'Villa';
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;
    submitButton.disabled = loading;

    const buttonText = submitButton.querySelector('.button-text');
    const buttonLoading = submitButton.querySelector('.button-loading');

    if (loading) {
        buttonText.classList.add('hidden');
        buttonLoading.classList.remove('hidden');
    } else {
        buttonText.classList.remove('hidden');
        buttonLoading.classList.add('hidden');
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
