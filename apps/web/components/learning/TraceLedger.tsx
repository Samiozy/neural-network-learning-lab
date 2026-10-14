"use client";
import React from "react";
import { useState } from "react";
import { Equation } from "../math/Equation";
import type { TraceDocument } from "@/lib/types";

export function TraceLedger({ trace }: { trace: TraceDocument | null }) {
  const [index, setIndex] = useState(0);
  if (!trace?.steps.length) return <section className="ledger panel"><p>Run a calculation to build its inspectable ledger.</p></section>;
  const step = trace.steps[Math.min(index, trace.steps.length - 1)];
  return <section className="ledger panel" aria-labelledby="ledger-title">
    <div className="panel-heading"><div><span className={`phase ${step.phase}`}>{step.phase}</span><h2 id="ledger-title">Calculation ledger</h2></div><span>Step {index + 1} / {trace.steps.length}</span></div>
    <ol className="step-dots" aria-label="Trace steps">{trace.steps.map((item, position) => <li key={item.step_id}><button aria-current={position === index ? "step" : undefined} aria-label={`Go to step ${position + 1}: ${item.operation}`} onClick={() => setIndex(position)}>{position + 1}</button></li>)}</ol>
    <h3>{step.operation.replaceAll("-", " ")}</h3><p>{step.description}</p>
    <div className="math-block"><span className="eyebrow">Symbolic formula</span><Equation latex={step.equation_latex} /></div>
    <div className="math-block"><span className="eyebrow">Numeric substitution</span><Equation latex={step.substitution_latex} /></div>
    <div className="shape-grid"><div><b>Inputs</b>{Object.entries(step.input_shapes).map(([name, shape]) => <code key={name}>{name}: ({shape.join(", ") || "scalar"})</code>)}</div><div><b>Outputs</b>{Object.entries(step.output_shapes).map(([name, shape]) => <code key={name}>{name}: ({shape.join(", ") || "scalar"})</code>)}</div></div>
    {step.derivation_notes?.map(note => <p className="note" key={note}>{note}</p>)}
    <details><summary>Exact values and code mapping</summary><pre>{JSON.stringify(step.values, null, 2)}</pre><code>{step.code_mapping}</code></details>
    <div className="controls"><button onClick={() => setIndex(i => Math.max(0, i - 1))} disabled={index === 0}>Previous</button><button onClick={() => setIndex(i => Math.min(trace.steps.length - 1, i + 1))} disabled={index === trace.steps.length - 1}>Next</button><button onClick={() => setIndex(0)}>Reset replay</button></div>
  </section>;
}
