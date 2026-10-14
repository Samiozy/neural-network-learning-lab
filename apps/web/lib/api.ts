import type { TraceDocument } from "./types";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export async function simulate(path: string, body: unknown): Promise<TraceDocument> {
  const response = await fetch(`${API}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) { const problem = await response.json().catch(() => ({})); throw new Error(problem.detail ?? `Simulation failed (${response.status})`); }
  return response.json() as Promise<TraceDocument>;
}

