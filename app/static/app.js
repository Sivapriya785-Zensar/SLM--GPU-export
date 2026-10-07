"use strict";

const messages = document.getElementById("messages");
const intro = document.getElementById("intro");
const composer = document.getElementById("composer");
const networkSelect = document.getElementById("network");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const settingsBtn = document.getElementById("settingsBtn");
const settingsOverlay = document.getElementById("settingsOverlay");
const settingsCloseBtn = document.getElementById("settingsCloseBtn");
const settingsSaveBtn = document.getElementById("settingsSaveBtn");
const modelSelect = document.getElementById("modelSelect");
const settingsStatus = document.getElementById("settingsStatus");

let conversationId = null;
let busy = false;
let transcript = [];
let currentSavedId = null;
const SAVED_KEY = "disputeSlm.savedChats";

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function hideIntro() {
  if (intro) intro.style.display = "none";
}

function addUserBubble(text, networkTag, record = true) {
  if (record) transcript.push({ k: "user", text, tag: networkTag || "" });
  hideIntro();
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="bubble">${networkTag ? `<span class="net-tag">${esc(networkTag)}</span><br>` : ""}${esc(text)}</div>`;
  messages.appendChild(row);
  scrollToBottom();
}

function addTyping() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.innerHTML = `<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>`;
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function addTextBubble(text, isError, record = true) {
  if (record) transcript.push({ k: "text", text, err: !!isError });
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.innerHTML = `<div class="bubble${isError ? " err-bubble" : ""}">${esc(text)}</div>`;
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function renderResult(d, record = true) {
  if (record) transcript.push({ k: "result", d });
  // Classification only — no resolution steps / evidence / advice. reason_label and
  // category come straight from the SLM's own answer (may be blank on an ML-only
  // fallback, where only the bare code is known).
  const engine = d.slm_used ? "model-classified" : "candidate-model fallback";
  // "Other" candidates = the support model's ranked list minus whichever code the
  // SLM actually picked, so this never just repeats the headline result.
  const cands = (d.classifier_candidates || [])
    .filter((c) => c.code !== d.reason_code)
    .map((c) => `<span class="cand">${esc(c.code)}${c.probability != null ? ` ${Math.round(c.probability * 100)}%` : ""}</span>`)
    .join("");

  // On an ML-only fallback there's no SLM-sourced label — say so plainly instead of
  // leaving the code badge with nothing next to it.
  const labelHtml = d.reason_label
    ? `<span class="code-label">${esc(d.reason_label)}</span>`
    : `<span class="code-label code-label-muted">Label unavailable (SLM did not respond)</span>`;

  const html = `
    <div class="result">
      <div class="code-line">
        <span class="code-badge">${esc(d.reason_code)}</span>
        ${labelHtml}
      </div>
      <div class="meta">${d.category ? esc(d.category) + " · " : ""}${esc(d.network)} &middot; ${esc(engine)}</div>
      <p class="explain">${esc(d.slm_explanation || "No explanation available for this classification.")}</p>
      ${cands ? `<div class="cands"><span class="cands-label">Other candidates:</span> ${cands}</div>` : ""}
    </div>`;

  const row = document.createElement("div");
  row.className = "msg assistant";
  row.innerHTML = `<div class="bubble">${html}</div>`;
  messages.appendChild(row);
  scrollToBottom();
}

async function loadNetworks() {
  try {
    const data = await (await fetch("/api/networks")).json();
    const names = Object.keys(data);
    for (const n of names) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = n;
      networkSelect.appendChild(opt);
    }
  } catch (err) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Unable to load networks";
    networkSelect.appendChild(opt);
  }
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function setBusy(v) {
  busy = v;
  sendBtn.disabled = v;
  input.disabled = v;
}

function lockNetwork() {
  networkSelect.disabled = true;
}

function resetConversation() {
  conversationId = null;
  transcript = [];
  currentSavedId = null;
  renderSaved();
  messages.innerHTML = "";
  const div = document.createElement("div");
  div.id = "intro";
  div.className = "intro";
  messages.appendChild(div);
  networkSelect.disabled = false;
  networkSelect.selectedIndex = 0;
  input.value = "";
  autoGrow();
  input.focus();
}

newChatBtn.addEventListener("click", resetConversation);

composer.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;

  const isFirstTurn = !conversationId;
  if (isFirstTurn && !networkSelect.value) {
    networkSelect.reportValidity();
    return;
  }

  input.value = "";
  autoGrow();
  setBusy(true);

  if (isFirstTurn) {
    addUserBubble(text, networkSelect.value);
    lockNetwork();
    const typing = addTyping();
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ narrative: text, network: networkSelect.value }),
      });
      typing.remove();
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      if (data.not_a_dispute) {
        // Not a real classification - don't start a conversation or lock the
        // network in, just explain and let the user try again immediately.
        networkSelect.disabled = false;
        addTextBubble(data.slm_explanation);
      } else {
        conversationId = data.conversation_id;
        renderResult(data);
      }
    } catch (err) {
      typing.remove();
      addTextBubble("Sorry, that request failed: " + err.message, true);
    } finally {
      setBusy(false);
      input.focus();
    }
  } else {
    addUserBubble(text);
    const typing = addTyping();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId, message: text }),
      });
      typing.remove();
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      addTextBubble(data.reply);
    } catch (err) {
      typing.remove();
      addTextBubble("Sorry, that request failed: " + err.message, true);
    } finally {
      setBusy(false);
      input.focus();
    }
  }
});

async function loadModels() {
  modelSelect.innerHTML = "";
  settingsStatus.textContent = "";
  settingsStatus.classList.remove("settings-status-err");
  try {
    const data = await (await fetch("/api/models")).json();
    const models = data.models || [];
    if (!models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No models found in Ollama";
      modelSelect.appendChild(opt);
      return;
    }
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      if (m.name === data.active) opt.selected = true;
      modelSelect.appendChild(opt);
    }
  } catch (err) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Unable to reach Ollama";
    modelSelect.appendChild(opt);
    settingsStatus.textContent = "Could not load models: " + err.message;
    settingsStatus.classList.add("settings-status-err");
  }
}

function openSettings() {
  settingsOverlay.hidden = false;
  loadModels();
}

function closeSettings() {
  settingsOverlay.hidden = true;
}

settingsBtn.addEventListener("click", openSettings);
settingsCloseBtn.addEventListener("click", closeSettings);
settingsOverlay.addEventListener("click", (ev) => {
  if (ev.target === settingsOverlay) closeSettings();
});

settingsSaveBtn.addEventListener("click", async () => {
  const chosen = modelSelect.value;
  if (!chosen) return;
  settingsSaveBtn.disabled = true;
  settingsStatus.textContent = "Switching model…";
  settingsStatus.classList.remove("settings-status-err");
  try {
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: chosen }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    closeSettings();
  } catch (err) {
    settingsStatus.textContent = "Failed to switch model: " + err.message;
    settingsStatus.classList.add("settings-status-err");
  } finally {
    settingsSaveBtn.disabled = false;
  }
});

const saveChatBtn = document.getElementById("saveChatBtn");
const savedList = document.getElementById("savedList");

function readSaved() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]");
  } catch (err) {
    return [];
  }
}

function writeSaved(list) {
  try {
    localStorage.setItem(SAVED_KEY, JSON.stringify(list));
    return true;
  } catch (err) {
    return false;
  }
}

function renderSaved() {
  const list = readSaved();
  savedList.innerHTML = "";
  if (!list.length) {
    savedList.innerHTML = `<li class="saved-empty">No saved chats yet.</li>`;
    return;
  }
  for (const c of list) {
    const li = document.createElement("li");
    li.className = "saved-item" + (c.id === currentSavedId ? " active" : "");
    li.innerHTML = `
      <button type="button" class="saved-open" title="${esc(c.title)}">${esc(c.title)}</button>
      <button type="button" class="saved-del" aria-label="Delete saved chat">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2">
          <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>`;
    li.querySelector(".saved-open").addEventListener("click", () => openSaved(c.id));
    li.querySelector(".saved-del").addEventListener("click", () => deleteSaved(c.id));
    savedList.appendChild(li);
  }
}

function saveCurrentChat() {
  if (!transcript.length) {
    addTextBubble("Nothing to save yet - send a message first.", false, false);
    return;
  }
  const first = transcript.find((t) => t.k === "user");
  const title = first ? first.text.slice(0, 40) + (first.text.length > 40 ? "…" : "") : "Chat";
  const entry = {
    id: currentSavedId || String(Date.now()),
    title,
    network: networkSelect.value,
    conversationId,
    transcript,
    savedAt: Date.now(),
  };
  const list = readSaved().filter((c) => c.id !== entry.id);
  list.unshift(entry);
  if (!writeSaved(list)) {
    addTextBubble("Could not save - browser storage is unavailable.", true, false);
    return;
  }
  currentSavedId = entry.id;
  renderSaved();
}

function openSaved(id) {
  const c = readSaved().find((x) => x.id === id);
  if (!c) return;
  messages.innerHTML = "";
  transcript = c.transcript.slice();
  conversationId = c.conversationId;
  currentSavedId = c.id;
  for (const t of transcript) {
    if (t.k === "user") addUserBubble(t.text, t.tag, false);
    else if (t.k === "text") addTextBubble(t.text, t.err, false);
    else if (t.k === "result") renderResult(t.d, false);
  }
  if (c.network) networkSelect.value = c.network;
  networkSelect.disabled = true;
  renderSaved();
}

function deleteSaved(id) {
  if (!confirm("Delete this saved chat?")) return;
  writeSaved(readSaved().filter((c) => c.id !== id));
  if (currentSavedId === id) currentSavedId = null;
  renderSaved();
}

saveChatBtn.addEventListener("click", saveCurrentChat);
renderSaved();

loadNetworks();
