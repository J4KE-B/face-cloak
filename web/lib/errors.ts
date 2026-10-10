export type ErrorKind = "network" | "timeout" | "aborted" | "http";

export class ApiError extends Error {
  kind: ErrorKind;
  status: number;
  retryAfter: number | null;
  detail: string | null;
  constructor(kind: ErrorKind, status: number, message: string, detail: string | null = null, retryAfter: number | null = null) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

