import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MathFoundations } from "../features/prerequisites/MathFoundations";

describe("MathFoundations", () => {
  it("exposes the complete prerequisite path and runs an editable dot product", async () => {
    const user = userEvent.setup();
    const run = vi.fn().mockResolvedValue(undefined);
    render(React.createElement(MathFoundations, { run, trace: null }));
    expect(screen.getByRole("button", { name: "1. Scalars & arithmetic" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "3. Dot products" }));
    await user.clear(screen.getByLabelText("Vector x"));
    await user.type(screen.getByLabelText("Vector x"), "1, 2");
    await user.clear(screen.getByLabelText("Vector y"));
    await user.type(screen.getByLabelText("Vector y"), "3, 4");
    await user.click(screen.getByRole("button", { name: "Build multiplication ledger" }));
    expect(run).toHaveBeenCalledWith("/api/math", { operation: "dot", left: [1, 2], right: [3, 4] });
  });
});
