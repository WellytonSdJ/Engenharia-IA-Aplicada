import type * as http from "node:http";
import express, { type Express } from "express";
import { chatErrorHandler, createChatHandler } from "../controllers/chat-controller.js";
import { resolveChatDependencies, type ChatDependencies } from "../services/chat.js";

export function createServer(deps?: Partial<ChatDependencies>): Express {
  const resolved = resolveChatDependencies(deps);
  const app = express();
  app.post("/chat", express.json(), createChatHandler(resolved));
  app.use(chatErrorHandler);
  return app;
}

export function startServer(port: number, deps?: Partial<ChatDependencies>): Promise<http.Server> {
  return new Promise((resolve) => {
    const server = createServer(deps).listen(port, () => {
      resolve(server);
    });
  });
}
