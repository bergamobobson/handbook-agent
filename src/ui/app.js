// ── Session & input ─────────────────────────────────────────────
const sessionId = 'session-' + Math.random().toString(36).slice(2, 8);
const input = document.getElementById('input');

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

// Send on Enter (Shift+Enter for new line)
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Markdown renderer ───────────────────────────────────────────
function inline(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>');
}

function renderText(text) {
  const lines = text.split('\n');
  const output = [];
  let inList = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (/^[-*] /.test(trimmed)) {
      if (!inList) { output.push('<ul>'); inList = true; }
      output.push(`<li>${inline(trimmed.slice(2))}</li>`);
    } else if (trimmed === '') {
      if (inList) { output.push('</ul>'); inList = false; }
    } else {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<p>${inline(trimmed)}</p>`);
    }
  }
  if (inList) output.push('</ul>');
  return output.join('');
}

// ── Append message bubble ───────────────────────────────────────
function appendMessage(role, content, source) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const messages = document.getElementById('messages');
  const wrap     = document.createElement('div');
  wrap.className = `message ${role}`;

  // Avatar
  const avatar = document.createElement('div');
  avatar.className = `avatar ${role === 'user' ? 'user' : 'bot'}`;
  avatar.textContent = role === 'user' ? 'You' : '🤖';

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  // Source tag for bot
  if (role === 'bot' && source) {
      const tag = document.createElement('div');
      
      // On ajoute la classe de base ET la classe spécifique (ex: source-tag handbook)
      tag.className = `source-tag ${source}`; 

      let icon = '';
      switch (source) {
        case 'handbook':
          icon = '📚';
          break;
        case 'conversational':
          icon = '💬';
          break;
        case 'off_topic':
          icon = '⚠️';
          break;
        default:
          icon = '🤖';
      }

      tag.textContent = `${icon} ${source.replace('_', ' ')}`;
      bubble.appendChild(tag);
  }

  // Text
  const text = document.createElement('div');
  text.innerHTML = renderText(content);
  bubble.appendChild(text);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

// ── Typing indicator ────────────────────────────────────────────
function showTyping() {
  const messages = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'typing';
  el.id = 'typing';
  el.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div class="typing-dots">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>`;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

// ── Send message ───────────────────────────────────────────────
async function sendMessage() {
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send').disabled = true;

  appendMessage('user', question);
  showTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, thread_id: sessionId }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    hideTyping();
    appendMessage('bot', data.answer, data.source);
  } catch (err) {
    hideTyping();
    appendMessage('bot', `⚠️ Error: ${err.message}. Is the API running?`);
  }

  document.getElementById('send').disabled = false;
  input.focus();
}

// ── Helpers ─────────────────────────────────────────────────────
function fillInput(text) {
  input.value = text;
  input.focus();
}

function sendSuggestion(btn) {
  input.value = btn.textContent.trim();
  sendMessage();
}