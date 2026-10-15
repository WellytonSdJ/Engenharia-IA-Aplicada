import type { ErrorRequestHandler, RequestHandler } from "express";
import { z } from "zod";
import type { ChatDependencies } from "../services/chat.js";
import { runChat } from "../services/chat.js";
import { ExecutionTimeoutError, UnknownConversationError, UnknownStrategyError } from "../services/errors.js";

export const ChatRequestSchema = z.object({
  message: z.string().trim().min(1),
  strategy: z.string().trim().min(1).optional(),
  reflect: z.boolean().default(false),
  conversationId: z.string().trim().min(1).optional(),
});

export function createChatHandler(deps: ChatDependencies): RequestHandler {
  return (req, res, next) => {
    ChatRequestSchema.parseAsync(req.body)
      .then((request) => runChat(request, deps))
      .then((result) => {
        res.status(200).json(result);
      })
      .catch(next);
  };
}

export function toErrorResponse(error: unknown): { status: number; body: Record<string, unknown> } {
  if (error instanceof z.ZodError) {
    return {
      status: 400,
      body: { error: "INVALID_BODY", message: "Request body is invalid.", issues: error.issues },
    };
  }
  if (error instanceof UnknownStrategyError) {
    return {
      status: 422,
      body: {
        error: "UNKNOWN_STRATEGY",
        message: error.message,
        strategy: error.strategy,
        available: error.available,
      },
    };
  }
  if (error instanceof ExecutionTimeoutError) {
    return { status: 504, body: { error: "TIMEOUT", message: error.message } };
  }
  if (error instanceof UnknownConversationError) {
    return {
      status: 422,
      body: {
        error: "UNKNOWN_CONVERSATION",
        message: error.message,
        conversationId: error.conversationId,
      },
    };
  }
  return { status: 500, body: { error: "INTERNAL", message: "Internal server error." } };
}

export const chatErrorHandler: ErrorRequestHandler = (error, _req, res, _next) => {
  if ((error as { type?: string })?.type === "entity.parse.failed") {
    res.status(400).json({
      error: "INVALID_BODY",
      message: "Request body is invalid.",
      issues: [{ code: "custom", path: [], message: "Malformed JSON body." }],
    });
    return;
  }

  const { status, body } = toErrorResponse(error);
  if (status === 500) {
    console.error(error);
  }
  res.status(status).json(body);
};
