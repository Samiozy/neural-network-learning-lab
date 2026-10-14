export function NetworkDiagram({ layers = [2, 3, 1], active = -1 }: { layers?: number[]; active?: number }) {
  const width = 480, height = 230; const gap = width / (layers.length + 1);
  const positions = layers.map((count, l) => Array.from({ length: count }, (_, n) => ({ x: gap * (l + 1), y: height * (n + 1) / (count + 1) })));
  return <svg className="network" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Dense network with layer sizes ${layers.join(", ")}`}>
    {positions.slice(0, -1).flatMap((layer, l) => layer.flatMap((source, s) => positions[l + 1].map((target, t) => <line key={`${l}-${s}-${t}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} />)))}
    {positions.flatMap((layer, l) => layer.map((point, n) => <g key={`${l}-${n}`}><circle className={active === l ? "active" : ""} cx={point.x} cy={point.y} r="20" /><text x={point.x} y={point.y + 5}>{l === 0 ? `x${n + 1}` : `a${l},${n + 1}`}</text></g>))}
  </svg>;
}

