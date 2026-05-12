const API = "/v1/chat/query";
const IMAGE_PROXY = "/v1/chat/image-proxy";
const PROGRESS_API = "/v1/chat/progress";

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
  return wrap;
}

function appendThinkingPanel(title = "思考过程") {
  const thread = $("thread");
  const wrap = document.createElement("div");
  wrap.className = "thinking-wrap";
  wrap.innerHTML = `
    <div class="thinking-card">
      <button type="button" class="thinking-toggle" aria-expanded="true">
        <span class="thinking-title">${escapeHtml(title)}</span>
        <span class="thinking-status" data-thinking-status>运行中</span>
        <span class="thinking-caret">▾</span>
      </button>
      <div class="thinking-body" data-thinking-body>
        <div class="progress-empty">正在初始化...</div>
      </div>
    </div>
  `;
  const toggle = wrap.querySelector(".thinking-toggle");
  const body = wrap.querySelector("[data-thinking-body]");
  toggle?.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") !== "false";
    toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
    if (body) body.style.display = expanded ? "none" : "";
    const caret = toggle.querySelector(".thinking-caret");
    if (caret) caret.textContent = expanded ? "▸" : "▾";
  });
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

function updateThinkingPanel(panel, payload) {
  if (!panel) return;
  const body = panel.querySelector("[data-thinking-body]");
  const statusEl = panel.querySelector("[data-thinking-status]");
  if (statusEl) {
    const statusMap = {
      running: "运行中",
      completed: "已完成",
      error: "失败",
      not_found: "等待中",
    };
    statusEl.textContent = statusMap[payload.status] || "运行中";
  }
  if (body) {
    body.innerHTML = renderProgress(payload.events || []);
    const latest = payload.events && payload.events.length ? payload.events[payload.events.length - 1] : null;
    body.dataset.lastTs = String(latest?.ts_ms || Date.now());
    body.dataset.lastStage = latest?.stage || "";
  }
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
    request_id: `req-${crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`}`,
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
  const data = await res.json();
  data.__request_id = body.request_id;
  return data;
}

function renderProgress(events) {
  if (!events || !events.length) return "<div class='progress-empty'>等待进度...</div>";
  const MIN_VISIBLE_STAGE_MS = 700;
  const sliced = events.slice(-24);
  const enriched = sliced.map((e, idx) => {
    const next = sliced[idx + 1];
    const currentTs = Number(e.ts_ms || 0);
    const nextTs = Number(next?.ts_ms || currentTs);
    const stageDurationMs = next ? Math.max(0, nextTs - currentTs) : null;
    return { ...e, stage_duration_ms: stageDurationMs };
  });

  const merged = [];
  let fastBuffer = [];
  for (const e of enriched) {
    const short =
      typeof e.stage_duration_ms === "number" &&
      e.stage_duration_ms < MIN_VISIBLE_STAGE_MS &&
      !e.data;
    if (short) {
      fastBuffer.push(e);
      continue;
    }
    if (fastBuffer.length) {
      const quick = fastBuffer.map((x) => x.stage).filter(Boolean).slice(0, 3);
      e._quickHint = `已快速完成 ${fastBuffer.length} 个阶段${quick.length ? `：${quick.join("、")}` : ""}`;
      fastBuffer = [];
    }
    merged.push(e);
  }
  if (!merged.length && fastBuffer.length) {
    merged.push(fastBuffer[fastBuffer.length - 1]);
  }

  const items = merged.slice(-16).map((e) => {
    const msg = escapeHtml(e.message || "");
    const stage = escapeHtml(e.stage || "");
    const t = e.ts_ms ? new Date(e.ts_ms).toLocaleTimeString() : "";
    const stageCost =
      typeof e.stage_duration_ms === "number" ? `${e.stage_duration_ms} ms` : "进行中";
    const quickHint = e._quickHint
      ? `<div class="progress-quick-hint">${escapeHtml(e._quickHint)}</div>`
      : "";
    const data =
      e.data && Object.keys(e.data).length
        ? `<div class="progress-data"><code>${escapeHtml(JSON.stringify(e.data, null, 2))}</code></div>`
        : "";
    return `<div class="progress-item"><div class="progress-stage">${stage}<span class="progress-time">${escapeHtml(t)} · ${escapeHtml(stageCost)}</span></div><div class="progress-msg">${msg}</div>${quickHint}${data}</div>`;
  }).join("");
  return `<div class="progress-block">${items}</div>`;
}

function appendProgressHeartbeat(panel) {
  if (!panel) return;
  const body = panel.querySelector("[data-thinking-body]");
  if (!body) return;
  const items = body.querySelectorAll(".progress-item");
  if (!items.length) return;
  items.forEach((el) => el.classList.remove("progress-wave"));
  const active = items[items.length - 1];
  active.classList.add("progress-wave");
  const now = Date.now();
  const lastTs = Number(body.dataset.lastTs || now);
  const sec = Math.max(1, Math.floor((now - lastTs) / 1000));
  const msg = active.querySelector(".progress-msg");
  if (!msg) return;
  let hint = active.querySelector(".progress-wave-hint");
  if (!hint) {
    hint = document.createElement("span");
    hint.className = "progress-wave-hint";
    msg.appendChild(document.createElement("br"));
    msg.appendChild(hint);
  }
  hint.textContent = `该阶段仍在执行（已等待 ${sec}s），系统正在处理，请稍候。`;
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

    let thinkingPanel = null;
    let progressStop = false;
    let pollTimer = null;
    let progressApiMissing = false;
    try {
      const previewReqId = `req-${crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`}`;
      thinkingPanel = appendThinkingPanel("思考过程（实时）");
      updateThinkingPanel(thinkingPanel, {
        status: "running",
        events: [
          {
            stage: "frontend.request_sent",
            message: "请求已发送，等待后端返回阶段信息。",
            data: { request_id: previewReqId },
            ts_ms: Date.now(),
          },
        ],
      });

      const bodyPromise = (async () => {
        const intentRaw = $("intentSelect").value;
        const intent = intentRaw === "" ? null : intentRaw;
        const urlVal = ($("urlInput").value || "").trim() || null;
        const body = {
          uid: currentUid,
          request_id: previewReqId,
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
        const data = await res.json();
        data.__request_id = previewReqId;
        return data;
      })();

      const pollOnce = async () => {
        if (progressStop) return;
        try {
          const pr = await fetch(
            `${PROGRESS_API}?request_id=${encodeURIComponent(previewReqId)}&_t=${Date.now()}`,
            { cache: "no-store" }
          );
          if (pr.status === 404) {
            progressApiMissing = true;
            updateThinkingPanel(thinkingPanel, {
              status: "running",
              events: [
                {
                  stage: "frontend.progress_unavailable",
                  message: "后端未暴露进度接口，实时阶段流不可用；请重启主服务后重试。",
                  data: { progress_api: PROGRESS_API, request_id: previewReqId },
                  ts_ms: Date.now(),
                },
              ],
            });
            return;
          }
          if (!pr.ok) return;
          const pd = await pr.json();
          const body = thinkingPanel?.querySelector("[data-thinking-body]");
          const beforeCount = body ? body.querySelectorAll(".progress-item").length : 0;
          updateThinkingPanel(thinkingPanel, pd);
          const afterCount = body ? body.querySelectorAll(".progress-item").length : 0;
          if (afterCount === beforeCount && pd.status === "running") {
            appendProgressHeartbeat(thinkingPanel);
          }
          if (pd.status === "completed" || pd.status === "error") {
            progressStop = true;
            return;
          }
        } catch {
          // ignore polling errors
        } finally {
          if (!progressStop) pollTimer = setTimeout(pollOnce, 350);
        }
      };
      pollTimer = setTimeout(pollOnce, 80);

      const data = await bodyPromise;
      progressStop = true;
      if (pollTimer) clearTimeout(pollTimer);
      if (progressApiMissing) {
        updateThinkingPanel(thinkingPanel, {
          status: "completed",
          events: [
            {
              stage: "frontend.fallback_done",
              message: "已返回最终答案（当前环境未启用实时进度流）。",
              data: { progress_api: PROGRESS_API },
              ts_ms: Date.now(),
            },
          ],
        });
      } else {
        // keep latest streamed events, only mark status completed
        const bodyNode = thinkingPanel?.querySelector("[data-thinking-body]");
        const eventsHtml = bodyNode?.innerHTML || renderProgress([]);
        const statusNode = thinkingPanel?.querySelector("[data-thinking-status]");
        if (statusNode) statusNode.textContent = "已完成";
        if (bodyNode) bodyNode.innerHTML = eventsHtml;
      }
      const answer = data.answer ?? "";
      const trace = data.trace_id ?? "";
      const route = data.route ?? "";
      const lat = data.latency_ms ?? "";
      const flags = data.runtime_flags ?? [];

      const meta = `<div class="msg-meta">trace <code>${escapeHtml(String(trace))}</code> - route <code>${escapeHtml(String(route))}</code> - ${escapeHtml(String(lat))} ms</div>`;
      const inner = `<div class="msg-role">助手</div><div class="msg-body">${escapeHtml(answer)}</div>${meta}${renderFlags(flags)}${renderEvidence(data.evidence)}${renderImages(data.images)}`;
      appendMessage("assistant", inner);
    } catch (err) {
      progressStop = true;
      if (pollTimer) clearTimeout(pollTimer);
      updateThinkingPanel(thinkingPanel, {
        status: "error",
        events: [
          {
            stage: "frontend.error",
            message: "请求失败。",
            data: { error: err.message || String(err) },
            ts_ms: Date.now(),
          },
        ],
      });
      showError(err.message || String(err));
    } finally {
      btn.disabled = false;
      ta.focus();
    }
  });

  autoResize(ta);
}

init();
