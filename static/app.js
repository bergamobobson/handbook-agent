const sessionId = 'session-' + Math.random().toString(36).slice(2, 8);

const input = document.getElementById('input');

// ── Auto-resize textarea ──────────────────────────────────────────────────────

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Markdown renderer ─────────────────────────────────────────────────────────
// Traite le texte ligne par ligne pour éviter les conflits de regex

function inline(text) {
  // Inline styles : bold, italic, code
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em>$1</em>')
    .replace(/`(.*?)`/g,       '<code>$1</code>');
}

function renderText(text) {
  const lines  = text.split('\n');
  const output = [];
  let inList   = false;

  for (const line of lines) {
    const trimmed = line.trim();

    // Ligne de liste  →  - item  ou  * item
    if (/^[-*] /.test(trimmed)) {
      if (!inList) { output.push('<ul>'); inList = true; }
      output.push(`<li>${inline(trimmed.slice(2))}</li>`);

    // Ligne vide  →  ferme la liste si ouverte, sinon séparateur
    } else if (trimmed === '') {
      if (inList) { output.push('</ul>'); inList = false; }

    // Ligne normale  →  ferme la liste si ouverte, puis paragraphe
    } else {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<p>${inline(trimmed)}</p>`);
    }
  }

  // Fermer la liste si le texte se termine sur un item
  if (inList) output.push('</ul>');

  return output.join('');
}

// ── Append a message bubble ───────────────────────────────────────────────────

function appendMessage(role, content, source) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const messages     = document.getElementById('messages');
  const wrap         = document.createElement('div');
  wrap.className     = `message ${role}`;

  const avatar       = document.createElement('div');
  avatar.className   = `avatar ${role === 'user' ? 'user' : 'bot'}`;
  avatar.textContent = role === 'user' ? 'You' : '🤖';

  const bubble     = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'bot' && source) {
    const tag       = document.createElement('div');
    tag.className   = `source-tag ${source}`;
    tag.textContent = source === 'handbook' ? '📚 handbook' : '🌐 web';
    bubble.appendChild(tag);
  }

  const text     = document.createElement('div');
  text.innerHTML = renderText(content);
  bubble.appendChild(text);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

// ── Typing indicator ──────────────────────────────────────────────────────────

function showTyping() {
  const messages = document.getElementById('messages');
  const el       = document.createElement('div');
  el.className   = 'typing';
  el.id          = 'typing';
  el.innerHTML   = `
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

// ── Send message ──────────────────────────────────────────────────────────────

async function sendMessage() {
  const question = input.value.trim();
  if (!question) return;

  input.value        = '';
  input.style.height = 'auto';
  document.getElementById('send').disabled = true;

  appendMessage('user', question);
  showTyping();

  try {
    const res = await fetch('/ask', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question, thread_id: sessionId }),
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

// ── Helpers ───────────────────────────────────────────────────────────────────

function fillInput(text) {
  input.value = text;
  input.focus();
}

function sendSuggestion(btn) {
  input.value = btn.textContent.trim();
  sendMessage();
}