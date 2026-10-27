import { planAndExecuteStrategy } from "./plan-and-execute.js";
import { reactStrategy } from "./react.js";
import { withReflection } from "./reflection.js";
import type { ReasoningStrategy } from "./strategy.js";
import { UnknownStrategyError } from "../services/errors.js";

export type StrategyRegistry = Readonly<Record<string, ReasoningStrategy>>;

export const strategyRegistry: StrategyRegistry = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
};

export const DEFAULT_STRATEGY = "react";

export function resolveStrategy(
  name: string | undefined,
  options: { reflect: boolean },
  registry: StrategyRegistry = strategyRegistry,
): ReasoningStrategy {
  const key = name ?? DEFAULT_STRATEGY;
  if (!Object.hasOwn(registry, key)) {
    throw new UnknownStrategyError(key, Object.keys(registry));
  }
  const base = registry[key];
  return options.reflect ? withReflection(base) : base;
}
