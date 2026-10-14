const terms = [
  ["Activation", "A neuron’s output after a nonlinear or identity transformation."], ["Bias", "A learned scalar added to a weighted sum."],
  ["Gradient", "Partial derivatives describing how loss changes locally."], ["Logit", "An unnormalized classification score."],
  ["Parameter", "A weight or bias learned during training."], ["Tensor", "A numeric array; its shape lists the size of every axis."]
];
export default function Glossary() { return <main className="reading"><Link href="/">← Laboratory</Link><h1>Mathematical glossary</h1><dl>{terms.map(([term, definition]) => <div key={term}><dt>{term}</dt><dd>{definition}</dd></div>)}</dl></main>; }
import Link from "next/link";

