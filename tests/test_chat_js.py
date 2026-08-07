"""Execute the browser's pure SSE/text rendering paths with Node.js."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

def test_browser_sse_parser_handles_utf8_chunk_boundaries_and_safe_text() -> None:
    node = shutil.which("node")
    assert node is not None, "브라우저 JavaScript 검증에는 Node.js가 필요합니다."

    chat_js = Path(__file__).parents[1] / "static" / "chat.js"
    harness = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

function stubElement() {
  const classes = new Set();
  const attributes = new Map();
  const element = {
    textContent: "",
    hidden: false,
    disabled: false,
    value: "",
    children: [],
    className: "",
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      remove(...names) { names.forEach((name) => classes.delete(name)); },
      toggle(name, force) {
        if (force === true) classes.add(name);
        else if (force === false) classes.delete(name);
        else if (classes.has(name)) classes.delete(name);
        else classes.add(name);
      },
      contains(name) { return classes.has(name); },
    },
    addEventListener() {},
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    removeAttribute(name) { attributes.delete(name); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    replaceChildren() { this.children = []; },
    append(...values) { this.children.push(...values); },
    focus() {},
  };
  return element;
}

const selectorElements = new Map();
globalThis.document = {
  querySelector(selector) {
    if (!selectorElements.has(selector)) selectorElements.set(selector, stubElement());
    return selectorElements.get(selector);
  },
  createElement() { return stubElement(); },
};
globalThis.window = { addEventListener() {} };
globalThis.localStorage = {
  getItem() { return null; },
  setItem() {},
  removeItem() {},
};

const source = fs.readFileSync(process.argv[1], "utf8")
  + `\n;globalThis.__chatTest = {
      parseSseFrame, readSseEvents, setBubbleText, streamQuestion,
      renderCandidates, setLoading, sendQuestion, startNewConversation,
      elements,
      setThreadId(value) { threadId = value; },
      setActiveRequest(value) { activeRequest = value; },
      state() { return { threadId, isLoading, activeRequest }; },
    };`;
vm.runInThisContext(source, { filename: process.argv[1] });

async function main() {
  const parsed = __chatTest.parseSseFrame(
    'event: delta\r\ndata: {"text":"한글\\n줄바꿈"}\r\n'
  );
  assert.deepStrictEqual(parsed, {
    event: "delta",
    data: { text: "한글\n줄바꿈" },
  });

  const payload = [
    'event: delta\n',
    'data: {"text":"가나다"}\n\n',
    'event: replace\n',
    'data: {"answer":"최종"}\n\n',
  ].join("");
  const bytes = new TextEncoder().encode(payload);
  const chunks = [];
  for (let index = 0; index < bytes.length; index += 2) {
    chunks.push(bytes.slice(index, index + 2));
  }
  let cursor = 0;
  let released = false;
  const response = {
    body: {
      getReader() {
        return {
          async read() {
            if (cursor >= chunks.length) return { done: true };
            return { value: chunks[cursor++], done: false };
          },
          releaseLock() { released = true; },
        };
      },
    },
  };
  const events = [];
  await __chatTest.readSseEvents(response, (event, data) => {
    events.push({ event, data });
  });
  assert.deepStrictEqual(events, [
    { event: "delta", data: { text: "가나다" } },
    { event: "replace", data: { answer: "최종" } },
  ]);
  assert.strictEqual(released, true);

  const bubble = stubElement();
  __chatTest.setBubbleText(bubble, '<img src=x onerror="alert(1)">');
  assert.strictEqual(bubble.textContent, '<img src=x onerror="alert(1)">');
  assert.strictEqual(Object.hasOwn(bubble, "innerHTML"), false);

  const streamPayload = [
    'event: status\ndata: {"stage":"답변을 작성하는 중"}\n\n',
    'event: delta\ndata: {"text":"잠정"}\n\n',
    'event: replace\ndata: {"answer":"검증된 대체"}\n\n',
    'event: done\ndata: {"answer":"체크포인트 최종","candidates":["성실성"]}\n\n',
  ].join("");

  function streamingResponse(text) {
    const encoded = new TextEncoder().encode(text);
    const parts = [];
    for (let index = 0; index < encoded.length; index += 3) {
      parts.push(encoded.slice(index, index + 3));
    }
    let position = 0;
    return {
      ok: true,
      body: {
        getReader() {
          return {
            async read() {
              if (position >= parts.length) return { done: true };
              return { value: parts[position++], done: false };
            },
            releaseLock() {},
          };
        },
      },
    };
  }

  const validThread = "00000000-0000-4000-8000-000000000001";
  __chatTest.setThreadId(validThread);
  globalThis.fetch = async () => streamingResponse(streamPayload);
  const streamingBubble = stubElement();
  streamingBubble.classList.add("streaming-bubble");
  __chatTest.elements.messageList.setAttribute("aria-live", "off");
  const done = await __chatTest.streamQuestion(
    "성실성",
    new AbortController(),
    streamingBubble,
  );
  assert.strictEqual(done.answer, "체크포인트 최종");
  assert.strictEqual(streamingBubble.textContent, "체크포인트 최종");
  assert.strictEqual(streamingBubble.classList.contains("streaming-bubble"), false);
  assert.strictEqual(__chatTest.elements.messageList.getAttribute("aria-live"), "polite");

  __chatTest.renderCandidates(done.candidates);
  assert.strictEqual(__chatTest.elements.candidatePanel.hidden, false);
  assert.strictEqual(__chatTest.elements.candidateList.children.length, 1);
  assert.strictEqual(__chatTest.elements.candidateList.children[0].textContent, "성실성");

  let duplicateFetches = 0;
  globalThis.fetch = async () => {
    duplicateFetches += 1;
    return streamingResponse(streamPayload);
  };
  __chatTest.setLoading(true);
  await __chatTest.sendQuestion("중복 질문");
  assert.strictEqual(duplicateFetches, 0);
  assert.strictEqual(__chatTest.elements.messageInput.disabled, true);
  assert.strictEqual(__chatTest.elements.messageList.getAttribute("aria-busy"), "true");
  __chatTest.setLoading(false);

  let aborted = false;
  __chatTest.setThreadId(validThread);
  __chatTest.setActiveRequest({ abort() { aborted = true; } });
  let requestIndex = 0;
  globalThis.fetch = async () => {
    requestIndex += 1;
    if (requestIndex === 1) {
      return {
        ok: false,
        async json() { return { detail: "새 대화 생성 실패" }; },
      };
    }
    return {
      ok: true,
      async json() {
        return {
          messages: [{ role: "assistant", content: "체크포인트 답변" }],
          candidates: [],
        };
      },
    };
  };
  await __chatTest.startNewConversation();
  assert.strictEqual(aborted, true);
  assert.strictEqual(requestIndex, 2);
  assert.strictEqual(__chatTest.state().activeRequest, null);
  assert.strictEqual(__chatTest.state().isLoading, false);
  assert.strictEqual(__chatTest.elements.errorBanner.hidden, false);
  assert.strictEqual(__chatTest.elements.errorBanner.textContent, "새 대화 생성 실패");
  const restoredRow = __chatTest.elements.messageList.children[0];
  const restoredBubble = restoredRow.children[0].children[1];
  assert.strictEqual(restoredBubble.textContent, "체크포인트 답변");
}

main().catch((error) => {
  process.stderr.write(String(error.stack || error));
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", harness, str(chat_js)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
