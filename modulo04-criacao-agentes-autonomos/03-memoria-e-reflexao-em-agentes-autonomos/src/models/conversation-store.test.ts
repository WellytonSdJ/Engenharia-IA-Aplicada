import assert from "node:assert/strict";
import test from "node:test";
import { UnknownConversationError } from "../services/errors.js";
import { ConversationStore } from "./conversation-store.js";

function store(): ConversationStore {
  return new ConversationStore(":memory:");
}

test("create generates a distinct id on each call", async () => {
  const db = store();
  const first = await db.create();
  const second = await db.create();

  assert.notEqual(first.id, second.id);
  assert.ok(first.id.startsWith("conversation-"));
  assert.ok(first.createdAt);
});

test("append writes messages in the given order and lastMessages returns them oldest to newest", async () => {
  const db = store();
  const conversation = await db.create();

  await db.append(conversation.id, [
    { role: "user", content: "primeira" },
    { role: "assistant", content: "resposta" },
  ]);
  await db.append(conversation.id, [{ role: "user", content: "segunda" }]);

  const messages = await db.lastMessages(conversation.id, 12);
  assert.deepEqual(messages.map((message) => message.content), ["primeira", "resposta", "segunda"]);
  assert.deepEqual(messages.map((message) => message.role), ["user", "assistant", "user"]);
  for (const message of messages) {
    assert.equal(message.conversationId, conversation.id);
  }
});

test("lastMessages returns all messages when there are fewer than the limit", async () => {
  const db = store();
  const conversation = await db.create();
  await db.append(conversation.id, [
    { role: "user", content: "a" },
    { role: "assistant", content: "b" },
  ]);

  const messages = await db.lastMessages(conversation.id, 12);
  assert.equal(messages.length, 2);
});

test("lastMessages returns only the most recent N messages, in order, when there are more than the limit", async () => {
  const db = store();
  const conversation = await db.create();
  for (let i = 0; i < 13; i += 1) {
    await db.append(conversation.id, [{ role: "user", content: `msg-${i}` }]);
  }

  const messages = await db.lastMessages(conversation.id, 12);
  assert.equal(messages.length, 12);
  assert.deepEqual(messages.map((message) => message.content), Array.from({ length: 12 }, (_, i) => `msg-${i + 1}`));
});

test("lastMessages returns an empty list for a conversation with no messages", async () => {
  const db = store();
  const conversation = await db.create();

  const messages = await db.lastMessages(conversation.id, 12);
  assert.deepEqual(messages, []);
});

test("append throws UnknownConversationError for a nonexistent conversation id", async () => {
  const db = store();

  await assert.rejects(
    db.append("conversation-missing", [{ role: "user", content: "oi" }]),
    (error: unknown) => error instanceof UnknownConversationError && error.conversationId === "conversation-missing",
  );
});

test("lastMessages throws UnknownConversationError for a nonexistent conversation id", async () => {
  const db = store();

  await assert.rejects(
    db.lastMessages("conversation-missing", 12),
    (error: unknown) => error instanceof UnknownConversationError,
  );
});
