export type DomainErrorCode =
  | "VALIDATION"
  | "NOT_FOUND"
  | "INVALID_TRANSITION"
  | "CONFIGURATION"
  | "ITERATION_LIMIT";

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
