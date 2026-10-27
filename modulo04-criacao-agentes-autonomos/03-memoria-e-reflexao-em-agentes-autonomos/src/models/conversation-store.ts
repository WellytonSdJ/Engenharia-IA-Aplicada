import { DatabaseSync, type StatementSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { ConversationSchema, MessageSchema, type Conversation, type Message, type MessageRole } from "./domain.js";
import { UnknownConversationError } from "../services/errors.js";

interface ConversationRow {
  id: string;
  created_at: string;
}

interface MessageRow {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
  seq: number;
}

const toConversation = (row: ConversationRow): Conversation =>
  ConversationSchema.parse({ id: row.id, createdAt: row.created_at });

const toMessage = (row: MessageRow): Message =>
  MessageSchema.parse({
    id: row.id,
    conversationId: row.conversation_id,
    role: row.role,
    content: row.content,
    createdAt: row.created_at,
  });

export interface NewMessage {
  role: MessageRole;
  content: string;
}

export class ConversationStore {
  private readonly db: DatabaseSync;

  constructor(path: string = process.env.OPSPILOT_DB ?? "./data/opspilot.db") {
    if (path !== ":memory:") {
      mkdirSync(dirname(path), { recursive: true });
    }
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS conversations (
        id         TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        seq             INTEGER NOT NULL
      );
    `);
  }

  async create(): Promise<Conversation> {
    const id = `conversation-${randomUUID()}`;
    const createdAt = new Date().toISOString();
    this.db.prepare("INSERT INTO conversations (id, created_at) VALUES (?, ?)").run(id, createdAt);
    return toConversation({ id, created_at: createdAt });
  }

  async append(conversationId: string, messages: NewMessage[]): Promise<void> {
    this.assertExists(conversationId);
    const nextSeq = this.nextSeq(conversationId);
    const insert = this.db.prepare(
      "INSERT INTO messages (id, conversation_id, role, content, created_at, seq) VALUES (?, ?, ?, ?, ?, ?)",
    );
    const createdAt = new Date().toISOString();
    messages.forEach((message, index) => {
      insert.run(`message-${randomUUID()}`, conversationId, message.role, message.content, createdAt, nextSeq + index);
    });
  }

  async lastMessages(conversationId: string, limit: number): Promise<Message[]> {
    this.assertExists(conversationId);
    const statement: StatementSync = this.db.prepare(
      "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
    );
    const rows = statement.all(conversationId, limit) as unknown as MessageRow[];
    return rows.reverse().map(toMessage);
  }

  private nextSeq(conversationId: string): number {
    const row = this.db
      .prepare("SELECT COALESCE(MAX(seq), -1) AS maxSeq FROM messages WHERE conversation_id = ?")
      .get(conversationId) as unknown as { maxSeq: number };
    return row.maxSeq + 1;
  }

  private assertExists(conversationId: string): void {
    const row = this.db.prepare("SELECT id FROM conversations WHERE id = ?").get(conversationId) as
      | { id: string }
      | undefined;
    if (!row) {
      throw new UnknownConversationError(conversationId);
    }
  }
}
