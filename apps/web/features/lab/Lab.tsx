"use client";
import { useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { TraceLedger } from "@/components/learning/TraceLedger";
import { NetworkDiagram } from "@/components/network/NetworkDiagram";
import { MathFoundations } from "@/features/prerequisites/MathFoundations";
import { simulate } from "@/lib/api";
import type { TraceDocument } from "@/lib/types";

const denseLayers = [
  { weights: [[0.2, -0.4], [0.7, 0.1], [-0.5, 0.8]], bias: [0.1, -0.2, 0.3], activation: "tanh" },
  { weights: [[0.6, -0.3, 0.5]], bias: [0.2], activation: "identity" }
];
const tabs = ["Math", "Neuron", "DNN forward", "Loss", "Backprop", "Training"] as const;
type Tab = typeof tabs[number];

function NumberField({ label, value, onChange, step = 0.1 }: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return <label>{label}<input type="number" step={step} value={value} onChange={e => onChange(Number(e.target.value))} /></label>;
}

export function Lab() {
  const [tab, setTab] = useState<Tab>("Math"); const [trace, setTrace] = useState<TraceDocument | null>(null); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const [x1, setX1] = useState(2); const [x2, setX2] = useState(1.5); const [w1, setW1] = useState(.4); const [w2, setW2] = useState(.7); const [bias, setBias] = useState(.1); const [activation, setActivation] = useState("relu");
  const [rate, setRate] = useState(.1); const [epochs, setEpochs] = useState(12); const [batch, setBatch] = useState(4); const [seed, setSeed] = useState(7);
  const run = async (path: string, body: unknown) => { setBusy(true); setError(""); try { setTrace(await simulate(path, body)); } catch (reason) { setError(reason instanceof Error ? reason.message : "Calculation failed"); } finally { setBusy(false); } };
  const select = (next: Tab) => { setTab(next); setTrace(null); setError(""); };
  const content = useMemo(() => {
    if (tab === "Math") return <MathFoundations run={run} trace={trace}/>;
    if (tab === "Neuron") return <><h2>Single-neuron microscope</h2><p>Inputs and weights are chosen values; weights and bias become learned parameters during training.</p><div className="field-grid"><NumberField label="x₁" value={x1} onChange={setX1}/><NumberField label="x₂" value={x2} onChange={setX2}/><NumberField label="w₁" value={w1} onChange={setW1}/><NumberField label="w₂" value={w2} onChange={setW2}/><NumberField label="bias" value={bias} onChange={setBias}/><label>Activation<select value={activation} onChange={e => setActivation(e.target.value)}><option>relu</option><option>sigmoid</option><option>tanh</option><option>identity</option></select></label></div><button onClick={() => run("/api/neuron", { x: [x1, x2], w: [w1, w2], b: bias, activation, seed })}>Recompute neuron</button><button className="secondary" onClick={() => { setX1(2); setX2(1.5); setW1(.4); setW2(.7); setBias(.1); setActivation("relu"); setTrace(null); }}>Restore preset</button></>;
    if (tab === "DNN forward") return <><h2>2 → 3 → 1 forward pass</h2><NetworkDiagram/><p>Switching ledger steps changes only the view; the canonical computed state stays intact.</p><button onClick={() => run("/api/dense", { x: [1, -2], layers: denseLayers, seed })}>Run deterministic forward pass</button></>;
    if (tab === "Loss") return <><h2>Predictions and loss</h2><p>MSE is for continuous targets. Stable BCE consumes binary logits; stable softmax cross-entropy consumes multiclass logits.</p><div className="button-row"><button onClick={() => run("/api/loss", { kind: "mse", prediction: [1.4], target: [2] })}>MSE</button><button onClick={() => run("/api/loss", { kind: "bce_logits", prediction: [1000], target: [1] })}>Stable BCE</button><button onClick={() => run("/api/loss", { kind: "softmax_ce", prediction: [2, 1, -1], target: 0 })}>Softmax + CE</button></div></>;
    if (tab === "Backprop") return <><h2>Backpropagation microscope</h2><NetworkDiagram/><p>Every parameter receives a derivative through the chain rule. This smooth tanh fixture can also be checked with central differences.</p><button onClick={() => run("/api/dense", { x: [1, -2], layers: denseLayers, target: [.5], seed })}>Forward → loss → backward</button><p className="challenge"><b>Check by hand:</b> predict the sign of the output bias gradient before opening its exact values.</p></>;
    return <><h2>Deterministic training</h2><p>The four-point dataset follows y = 2x + 1. Choose full-batch, mini-batch, or SGD and inspect every real update.</p><div className="field-grid"><NumberField label="Learning rate η" value={rate} onChange={setRate} step={.01}/><NumberField label="Epochs" value={epochs} onChange={setEpochs} step={1}/><label>Batch mode<select value={batch} onChange={e => setBatch(Number(e.target.value))}><option value="4">Full batch (4)</option><option value="2">Mini-batch (2)</option><option value="1">SGD (1)</option></select></label><NumberField label="Seed" value={seed} onChange={setSeed} step={1}/></div><button onClick={() => run("/api/training", { learning_rate: rate, epochs, batch_size: batch, seed })}>Train and trace</button>{trace?.snapshots && <div className="chart" aria-label="Loss across parameter updates"><ResponsiveContainer width="100%" height={240}><LineChart data={trace.snapshots}><XAxis dataKey="epoch"/><YAxis/><Tooltip/><Line type="monotone" dataKey="loss" stroke="#e8674b" dot={false}/></LineChart></ResponsiveContainer></div>}</>;
  }, [tab, x1, x2, w1, w2, bias, activation, rate, epochs, batch, seed, trace]);
  return <div className="lab-layout"><section className="workspace panel"><div className="tabs" role="tablist" aria-label="MVP learning labs">{tabs.map(item => <button role="tab" aria-selected={tab === item} key={item} onClick={() => select(item)}>{item}</button>)}</div>{content}{busy && <p role="status">Computing canonical trace…</p>}{error && <p className="error" role="alert">{error}</p>}</section><TraceLedger key={trace?.simulation_id ?? "empty"} trace={trace}/></div>;
}
