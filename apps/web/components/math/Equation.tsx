"use client";
import React from "react";
import katex from "katex";
export function Equation({ latex, label }: { latex: string; label?: string }) {
  let html = "";
  try { html = katex.renderToString(latex, { throwOnError: false, output: "htmlAndMathml" }); } catch { html = latex; }
  return <div className="equation" aria-label={label ?? latex} dangerouslySetInnerHTML={{ __html: html }} />;
}
