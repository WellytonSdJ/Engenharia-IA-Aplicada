import type { RequestHandler } from "express";
import { z } from "zod";
import type { MemoryStore } from "../models/memory-store.js";

export const RememberRequestSchema = z.object({
  userId: z.string().trim().min(1),
  fact: z.string().trim().min(1),
});

/**
 * POST /memories — grava um fato na memória semântica do usuário "na mão" (spec 009).
 * 201 = fato novo gravado; 200 = já existia um fato quase idêntico (dedup), devolve o id dele.
 * Erros caem no mesmo error handler do /chat (400 corpo inválido, 503 modelo indisponível).
 */
export function createRememberHandler(memoryStore: MemoryStore): RequestHandler {
  return (req, res, next) => {
    RememberRequestSchema.parseAsync(req.body)
      .then(async ({ userId, fact }) => {
        const result = await memoryStore.remember(userId, fact);
        res.status(result.stored ? 201 : 200).json({ ...result, userId, fact });
      })
      .catch(next);
  };
}
