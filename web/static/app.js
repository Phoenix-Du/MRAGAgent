const API = "/v1/chat/query";
const IMAGE_PROXY = "/v1/chat/image-proxy";

function uid() {
  return `web-${crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`}`;
}

let currentUid = uid();

const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
}

function renderFlags(flags) {
  if (!flags || !flags.length) return "";
  return `<div class="flags">${flags.map((f) => `<span class="flag">${escapeHtml(f)}</span>`).join("")}</div>`;
}

function renderEvidence(evidence) {
  if (!evidence || !evidence.length) return "";
  const items = evidence
    .map((e) => {
      const sn = escapeHtml((e.snippet || "").slice(0, 280));
      const sc =
        e.score != null && !Number.isNaN(Number(e.score))
          ? Number(e.score).toFixed(2)
          : "-";
      return `<div class="evidence-item"><strong>${escapeHtml(e.doc_id || "")}</strong> - score ${sc}<br/>${sn}</div>`;
    })
    .join("");
  return `<div class="evidence-block"><div class="evidence-title">引用片段</div>${items}</div>`;
}

function safeImageUrl(u) {
  if (!u || typeof u !== "string") return null;
  try {
    const url = new URL(u, window.location.origin);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url.href;
  } catch {
    return null;
  }
}

function renderImages(images) {
  if (!images || !images.length) return "";
  const imgs = images
    .map((im) => {
      const alt = escapeHtml(im.desc || "image");
      let proxied = "";
      if (im.local_path && typeof im.local_path === "string") {
        proxied = `${IMAGE_PROXY}?local_path=${encodeURIComponent(im.local_path)}`;
      } else {
        const href = safeImageUrl(im.url);
        if (!href) return "";
        proxied = `${IMAGE_PROXY}?url=${encodeURIComponent(href)}`;
      }
      return `<img src="${proxied}" alt="${alt}" loading="lazy" />`;
    })
    .join("");
  if (!imgs) return "";
  return `<div class="images-row">${imgs}</div>`;
}

function appendMessage(role, htmlInner) {
  const thread = $("thread");
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role === "user" ? "user" : "assistant"}`;
  wrap.innerHTML = `<div class="msg-bubble">${htmlInner}</div>`;
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
}

function showError(msg) {
  const thread = $("thread");
  const el = document.createElement("div");
  el.className = "error-toast";
  el.textContent = msg;
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
}

async function sendMessage(text) {
  const intentRaw = $("intentSelect").value;
  const intent = intentRaw === "" ? null : intentRaw;
  const urlVal = ($("urlInput").value || "").trim() || null;

  const body = {
    uid: currentUid,
    query: text,
    use_rasa_intent: true,
    intent_confidence_threshold: 0.6,
    max_images: 5,
    max_web_docs: 5,
  };
  if (intent) body.intent = intent;
  if (urlVal) body.url = urlVal;

  const res = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || `HTTP ${res.status}`);
  }
  return res.json();
}

function init() {
  $("uidDisplay").textContent = currentUid;

  $("btnNewChat").addEventListener("click", () => {
    currentUid = uid();
    $("uidDisplay").textContent = currentUid;
    $("thread").innerHTML = "";
    $("queryInput").value = "";
    $("urlInput").value = "";
    autoResize($("queryInput"));
  });

  const ta = $("queryInput");
  ta.addEventListener("input", () => autoResize(ta));

  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chatForm").requestSubmit();
    }
  });

  $("chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = ta.value.trim();
    if (!text) return;

    const btn = $("btnSend");
    btn.disabled = true;

    appendMessage(
      "user",
      `<div class="msg-role">你</div><div class="msg-body">${escapeHtml(text)}</div>`
    );
    ta.value = "";
    autoResize(ta);

    try {
      const data = await sendMessage(text);
      const answer = data.answer ?? "";
      const trace = data.trace_id ?? "";
      const route = data.route ?? "";
      const lat = data.latency_ms ?? "";
      const flags = data.runtime_flags ?? [];

      const meta = `<div class="msg-meta">trace <code>${escapeHtml(String(trace))}</code> - route <code>${escapeHtml(String(route))}</code> - ${escapeHtml(String(lat))} ms</div>`;
      const inner = `<div class="msg-role">助手</div><div class="msg-body">${escapeHtml(answer)}</div>${meta}${renderFlags(flags)}${renderEvidence(data.evidence)}${renderImages(data.images)}`;
      appendMessage("assistant", inner);
    } catch (err) {
      showError(err.message || String(err));
    } finally {
      btn.disabled = false;
      ta.focus();
    }
  });

  autoResize(ta);
}

init();
