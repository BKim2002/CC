"use strict";

const STORAGE_KEY = "competency_chat_thread_id";
const MAX_MESSAGE_LENGTH = 2000;

const elements = {
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newThreadButton: document.querySelector("#newThreadButton"),
  messageList: document.querySelector("#messageList"),
  candidatePanel: document.querySelector("#candidatePanel"),
  candidateList: document.querySelector("#candidateList"),
  statusText: document.querySelector("#statusText"),
  statusDot: document.querySelector("#statusDot"),
  errorBanner: document.querySelector("#errorBanner"),
  liveAnnouncement: document.querySelector("#liveAnnouncement"),
};

let threadId = null;
let isLoading = false;
let loadingMessage = null;


function isUuid(value) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value || "",
  );
}


function setStatus(message, state = "ready") {
  elements.statusText.textContent = message;
  elements.statusDot.className = `status-dot ${state}`;
}


function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
  elements.liveAnnouncement.textContent = message;
  setStatus("문제가 발생했어요", "error");
}


function clearError() {
  elements.errorBanner.textContent = "";
  elements.errorBanner.hidden = true;
}


function setLoading(loading) {
  isLoading = loading;
  elements.messageInput.disabled = loading;
  elements.sendButton.disabled = loading;
  elements.newThreadButton.disabled = loading;
  elements.messageList.setAttribute("aria-busy", String(loading));

  for (const button of elements.candidateList.querySelectorAll("button")) {
    button.disabled = loading;
  }

  if (loading) {
    setStatus("답변을 준비하고 있어요", "loading");
  } else if (!elements.errorBanner.hidden) {
    setStatus("문제가 발생했어요", "error");
  } else {
    setStatus("질문할 준비가 되었어요", "ready");
  }
}


function scrollToLatest() {
  elements.messageList.scrollTop = elements.messageList.scrollHeight;
}


function clearConversation() {
  elements.messageList.replaceChildren();
  renderCandidates([]);
}


function showWelcome() {
  if (elements.messageList.children.length > 0) {
    return;
  }

  const card = document.createElement("section");
  card.className = "welcome-card";

  const title = document.createElement("h2");
  title.textContent = "어떤 역량이 궁금하신가요?";

  const description = document.createElement("p");
  description.textContent =
    "‘성실성의 하위요인을 알려줘’처럼 묻거나, 찾고 싶은 행동의 특징을 자연스럽게 설명해 보세요.";

  card.append(title, description);
  elements.messageList.append(card);
}


function removeWelcome() {
  const welcome = elements.messageList.querySelector(".welcome-card");

  if (welcome) {
    welcome.remove();
  }
}


function appendMessage(role, content) {
  removeWelcome();

  const row = document.createElement("article");
  row.className = `message-row ${role}`;

  const group = document.createElement("div");
  group.className = "message-group";

  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "나" : "역량 챗봇";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;

  group.append(label, bubble);
  row.append(group);
  elements.messageList.append(row);
  scrollToLatest();

  return row;
}


function appendLoadingMessage() {
  const row = appendMessage("assistant", "");
  const bubble = row.querySelector(".message-bubble");
  bubble.classList.add("loading-bubble");
  bubble.setAttribute("aria-label", "답변을 준비하고 있습니다");

  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement("span");
    dot.className = "typing-dot";
    dot.setAttribute("aria-hidden", "true");
    bubble.append(dot);
  }

  loadingMessage = row;
}


function removeLoadingMessage() {
  if (loadingMessage) {
    loadingMessage.remove();
    loadingMessage = null;
  }
}


function renderCandidates(candidates) {
  elements.candidateList.replaceChildren();

  if (!Array.isArray(candidates) || candidates.length === 0) {
    elements.candidatePanel.hidden = true;
    return;
  }

  for (const candidate of candidates) {
    if (typeof candidate !== "string" || !candidate.trim()) {
      continue;
    }

    const button = document.createElement("button");
    button.className = "candidate-button";
    button.type = "button";
    button.textContent = candidate;
    button.disabled = isLoading;
    button.addEventListener("click", () => {
      sendQuestion(candidate);
    });

    elements.candidateList.append(button);
  }

  elements.candidatePanel.hidden = elements.candidateList.children.length === 0;
}


function validationMessage(detail) {
  if (!Array.isArray(detail)) {
    return typeof detail === "string" ? detail : null;
  }

  const messages = detail
    .map((item) => item?.msg)
    .filter((item) => typeof item === "string");

  return messages.length > 0 ? messages.join(" ") : null;
}


async function requestJson(url, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error("서버에 연결할 수 없습니다. 서버 실행 상태를 확인해 주세요.", {
      cause: error,
    });
  }

  let data;

  try {
    data = await response.json();
  } catch (error) {
    throw new Error("서버 응답을 읽지 못했습니다. 잠시 후 다시 시도해 주세요.", {
      cause: error,
    });
  }

  if (!response.ok) {
    const message = validationMessage(data.detail);
    throw new Error(message || "요청을 처리하지 못했습니다.");
  }

  return data;
}


async function createThread() {
  const data = await requestJson("/api/threads", {
    method: "POST",
  });

  if (!isUuid(data.thread_id)) {
    throw new Error("서버가 올바른 대화 ID를 반환하지 않았습니다.");
  }

  threadId = data.thread_id;
  localStorage.setItem(STORAGE_KEY, threadId);

  return threadId;
}


async function ensureThread() {
  const savedThreadId = localStorage.getItem(STORAGE_KEY);

  if (isUuid(savedThreadId)) {
    threadId = savedThreadId;
    return threadId;
  }

  localStorage.removeItem(STORAGE_KEY);
  return createThread();
}


async function restoreConversation() {
  const data = await requestJson(
    `/api/threads/${encodeURIComponent(threadId)}/messages`,
  );

  clearConversation();

  for (const message of data.messages || []) {
    if (message.role !== "user" && message.role !== "assistant") {
      continue;
    }

    appendMessage(message.role, String(message.content || ""));
  }

  renderCandidates(data.candidates || []);
  showWelcome();
}


async function sendQuestion(rawQuestion) {
  if (isLoading) {
    return;
  }

  const question = String(rawQuestion || "").trim();

  if (!question) {
    showError("질문을 입력해 주세요.");
    elements.messageInput.focus();
    return;
  }

  if (question.length > MAX_MESSAGE_LENGTH) {
    showError(`질문은 ${MAX_MESSAGE_LENGTH}자 이하로 입력해 주세요.`);
    elements.messageInput.focus();
    return;
  }

  clearError();
  setLoading(true);

  try {
    if (!isUuid(threadId)) {
      await ensureThread();
    }

    appendMessage("user", question);
    renderCandidates([]);
    elements.messageInput.value = "";
    appendLoadingMessage();

    const data = await requestJson("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: question,
        thread_id: threadId,
      }),
    });

    removeLoadingMessage();
    appendMessage("assistant", data.answer);
    renderCandidates(data.candidates || []);
    elements.liveAnnouncement.textContent = "새 답변이 도착했습니다.";
  } catch (error) {
    removeLoadingMessage();

    if (isUuid(threadId)) {
      try {
        await restoreConversation();
      } catch {
        // 서버 상태도 읽지 못하면 현재 화면을 유지하고 원래 오류를 알린다.
      }
    }

    showError(error.message || "알 수 없는 오류가 발생했습니다.");
  } finally {
    setLoading(false);
    elements.messageInput.focus();
    scrollToLatest();
  }
}


async function startNewConversation() {
  if (isLoading) {
    return;
  }

  clearError();
  setLoading(true);

  try {
    await createThread();
    clearConversation();
    showWelcome();
    elements.liveAnnouncement.textContent = "새 대화를 시작했습니다.";
  } catch (error) {
    showError(error.message || "새 대화를 시작하지 못했습니다.");
  } finally {
    setLoading(false);
    elements.messageInput.focus();
  }
}


async function initialize() {
  clearError();
  setLoading(true);
  setStatus("저장된 대화를 확인하고 있어요", "loading");

  try {
    await ensureThread();
    await restoreConversation();
  } catch (error) {
    showError(error.message || "대화를 준비하지 못했습니다.");
    showWelcome();
  } finally {
    setLoading(false);
    elements.messageInput.focus();
  }
}


elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(elements.messageInput.value);
});


elements.messageInput.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter"
    && !event.shiftKey
    && !event.isComposing
  ) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});


elements.newThreadButton.addEventListener("click", startNewConversation);

window.addEventListener("DOMContentLoaded", initialize);
