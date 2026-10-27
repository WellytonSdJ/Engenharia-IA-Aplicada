export type DomainErrorCode =
  | "VALIDATION"
  | "NOT_FOUND"
  | "INVALID_TRANSITION"
  | "CONFIGURATION"
  | "ITERATION_LIMIT"
  | "UNKNOWN_STRATEGY"
  | "TIMEOUT"
  | "UNKNOWN_CONVERSATION";

export class OpsPilotError extends Error {
  constructor(
    message: string,
    readonly code: DomainErrorCode,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class ToolBoundaryError extends OpsPilotError {
  constructor(code: DomainErrorCode, message: string, options?: ErrorOptions) {
    super(message, code, options);
  }
}

export class ValidationError extends OpsPilotError {
  constructor(message: string, options?: ErrorOptions) {
    super(message, "VALIDATION", options);
  }
}

export class ResourceNotFoundError extends OpsPilotError {
  constructor(message: string) {
    super(message, "NOT_FOUND");
  }
}

export class InvalidTransitionError extends OpsPilotError {
  constructor(message: string) {
    super(message, "INVALID_TRANSITION");
  }
}

export class AgentConfigurationError extends OpsPilotError {
  constructor(message: string) {
    super(message, "CONFIGURATION");
  }
}

export class IterationLimitError extends OpsPilotError {
  constructor(maxIterations: number) {
    super(`Strategy exceeded the limit of ${maxIterations} iterations.`, "ITERATION_LIMIT");
  }
}

export class UnknownStrategyError extends OpsPilotError {
  constructor(
    readonly strategy: string,
    readonly available: string[],
  ) {
    super(`Unknown strategy "${strategy}".`, "UNKNOWN_STRATEGY");
  }
}

export class ExecutionTimeoutError extends OpsPilotError {
  constructor(readonly timeoutMs: number) {
    super(`Strategy execution exceeded ${timeoutMs} ms.`, "TIMEOUT");
  }
}

export class UnknownConversationError extends OpsPilotError {
  constructor(readonly conversationId: string) {
    super(`Unknown conversation "${conversationId}".`, "UNKNOWN_CONVERSATION");
  }
}
