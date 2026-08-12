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
let activeRequest = null;


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
  // 답변 생성 중에는 이 버튼이 현재 요청을 취소하고 새 대화를 만든다.
  elements.newThreadButton.disabled = loading && activeRequest === null;
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
  setBubbleText(bubble, content);

  group.append(label, bubble);
  row.append(group);
  elements.messageList.append(row);
  scrollToLatest();

  return row;
}


function setBubbleText(bubble, content) {
  const text = String(content || "");
  bubble.textContent = text;
  bubble.classList.toggle(
    "tree-content",
    /(^|\n)\s*[├└│]/u.test(text),
  );
}


function appendStreamingMessage() {
  const row = appendMessage("assistant", "");
  const bubble = row.querySelector(".message-bubble");
  bubble.classList.add("streaming-bubble");
  bubble.setAttribute("aria-label", "답변을 작성하고 있습니다");
  elements.messageList.setAttribute("aria-live", "off");

  return { row, bubble };
}


function finishStreamingMessage(bubble, content) {
  setBubbleText(bubble, content);
  bubble.classList.remove("streaming-bubble");
  bubble.removeAttribute("aria-label");
  elements.messageList.setAttribute("aria-live", "polite");
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


function parseSseFrame(frame) {
  let eventName = "message";
  const dataLines = [];

  for (const line of frame.split(/\r\n|\n|\r/u)) {
    if (!line || line.startsWith(":")) {
      continue;
    }

    const separatorIndex = line.indexOf(":");
    const field = separatorIndex < 0 ? line : line.slice(0, separatorIndex);
    let value = separatorIndex < 0 ? "" : line.slice(separatorIndex + 1);

    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    if (field === "event") {
      eventName = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  let data;

  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch (error) {
    throw new Error("서버의 실시간 응답 형식을 읽지 못했습니다.", {
      cause: error,
    });
  }

  return { event: eventName, data };
}


async function readSseEvents(response, handleEvent) {
  if (!response.body) {
    throw new Error("이 브라우저에서는 실시간 답변을 받을 수 없습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const consumeFrames = (flush = false) => {
    while (true) {
      const boundary = /\r\n\r\n|\n\n|\r\r/u.exec(buffer);

      if (!boundary) {
        break;
      }

      const frame = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      const parsed = parseSseFrame(frame);

      if (parsed) {
        handleEvent(parsed.event, parsed.data);
      }
    }

    if (flush && buffer.trim()) {
      const parsed = parseSseFrame(buffer);
      buffer = "";

      if (parsed) {
        handleEvent(parsed.event, parsed.data);
      }
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        buffer += decoder.decode();
        consumeFrames(true);
        return;
      }

      // stream:true가 한글 UTF-8 문자의 임의 byte chunk 경계를 보존한다.
      buffer += decoder.decode(value, { stream: true });
      consumeFrames();
    }
  } finally {
    reader.releaseLock();
  }
}


async function responseError(response) {
  let data = null;

  try {
    data = await response.json();
  } catch {
    // JSON이 아닌 proxy 오류도 아래의 일반 메시지로 안전하게 처리한다.
  }

  return new Error(
    validationMessage(data?.detail) || "요청을 처리하지 못했습니다.",
  );
}


async function streamQuestion(question, controller, bubble) {
  let response;

  try {
    response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: question,
        thread_id: threadId,
      }),
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    throw new Error(
      "서버에 연결할 수 없습니다. 서버 실행 상태를 확인해 주세요.",
      { cause: error },
    );
  }

  if (!response.ok) {
    throw await responseError(response);
  }

  let visibleAnswer = "";
  let doneData = null;
  let streamError = null;

  await readSseEvents(response, (eventName, data) => {
    if (!data || typeof data !== "object") {
      return;
    }

    if (eventName === "status" && typeof data.stage === "string") {
      setStatus(data.stage, "loading");
      return;
    }

    if (eventName === "delta" && typeof data.text === "string") {
      visibleAnswer += data.text;
      setBubbleText(bubble, visibleAnswer);
      scrollToLatest();
      return;
    }

    if (eventName === "replace" && typeof data.answer === "string") {
      visibleAnswer = data.answer;
      setBubbleText(bubble, visibleAnswer);
      scrollToLatest();
      return;
    }

    if (eventName === "done" && typeof data.answer === "string") {
      visibleAnswer = data.answer;
      doneData = data;
      return;
    }

    if (eventName === "error") {
      // 오류가 확정되면 잠정 표시를 즉시 비워 checkpoint 복원 전에도 남기지 않는다.
      visibleAnswer = "";
      setBubbleText(bubble, "");
      streamError = new Error(
        typeof data.message === "string"
          ? data.message
          : "실시간 답변을 완료하지 못했습니다. 다시 시도해 주세요.",
      );
    }
  });

  if (streamError) {
    throw streamError;
  }

  if (!doneData) {
    throw new Error("실시간 연결이 일찍 종료되었습니다. 다시 시도해 주세요.");
  }

  finishStreamingMessage(bubble, doneData.answer);
  return doneData;
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
  const controller = new AbortController();
  activeRequest = controller;
  setLoading(true);
  let streamMessage = null;

  try {
    if (!isUuid(threadId)) {
      await ensureThread();
    }

    appendMessage("user", question);
    renderCandidates([]);
    elements.messageInput.value = "";
    streamMessage = appendStreamingMessage();

    const data = await streamQuestion(
      question,
      controller,
      streamMessage.bubble,
    );

    renderCandidates(data.candidates || []);
    elements.liveAnnouncement.textContent = (
      `새 답변이 도착했습니다. ${data.answer}`
    );
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }

    if (streamMessage?.row?.isConnected) {
      streamMessage.row.remove();
    }

    if (isUuid(threadId)) {
      try {
        await restoreConversation();
      } catch {
        // 서버 상태도 읽지 못하면 현재 화면을 유지하고 원래 오류를 알린다.
      }
    }

    showError(error.message || "알 수 없는 오류가 발생했습니다.");
  } finally {
    elements.messageList.setAttribute("aria-live", "polite");

    if (activeRequest === controller) {
      activeRequest = null;
      setLoading(false);
      elements.messageInput.focus();
      scrollToLatest();
    }
  }
}


async function startNewConversation() {
  if (activeRequest) {
    const controller = activeRequest;
    activeRequest = null;
    controller.abort();
  }

  clearError();
  setLoading(true);

  try {
    await createThread();
    clearConversation();
    showWelcome();
    elements.liveAnnouncement.textContent = "새 대화를 시작했습니다.";
  } catch (error) {
    if (isUuid(threadId)) {
      try {
        // If the previous stream was aborted after provisional deltas, restore
        // only checkpointed messages instead of leaving the partial bubble.
        await restoreConversation();
      } catch {
        const partialBubble = elements.messageList.querySelector(
          ".message-row.assistant .streaming-bubble",
        );
        partialBubble?.closest(".message-row")?.remove();
      }
    }

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
