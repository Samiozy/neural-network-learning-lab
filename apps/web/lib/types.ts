export type TracePhase = "forward" | "loss" | "backward" | "update";
export interface TraceStep {
  step_id: string; phase: TracePhase; operation: string; description: string;
  equation_latex: string; substitution_latex: string; input_refs: string[]; output_refs: string[];
  input_shapes: Record<string, number[]>; output_shapes: Record<string, number[]>;
  values: Record<string, unknown>; highlight_targets: string[]; derivation_notes: string[]; code_mapping: string;
}
export interface TraceDocument {
  trace_version: string; simulation_id: string; architecture: string; seed: number;
  precision_policy: { internal: string; display_decimals: number; atol: number };
  config: Record<string, unknown>; steps: TraceStep[]; initial_parameters: Record<string, unknown>;
  gradients: Record<string, unknown>; final_parameters: Record<string, unknown>; warnings: string[]; errors: string[];
  output?: number[]; loss?: number; parameter_count?: number; snapshots?: Record<string, number>[];
}

