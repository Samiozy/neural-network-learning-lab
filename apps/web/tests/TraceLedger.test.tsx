import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { TraceLedger } from "../components/learning/TraceLedger";
const step = (id:string) => ({ step_id:id, phase:"forward" as const, operation:id, description:"derived", equation_latex:"a=b", substitution_latex:"1=1", input_refs:[], output_refs:[], input_shapes:{}, output_shapes:{}, values:{}, highlight_targets:[], derivation_notes:[], code_mapping:"a = b" });
const trace = { trace_version:"1", simulation_id:"x", architecture:"test", seed:0, precision_policy:{internal:"float64",display_decimals:4,atol:1e-6}, config:{}, steps:[step("first"),step("second")], initial_parameters:{},gradients:{},final_parameters:{},warnings:[],errors:[] };
describe("TraceLedger", () => { it("uses stored steps for previous and next", async () => { const user=userEvent.setup(); render(React.createElement(TraceLedger, { trace })); expect(screen.getByText("first")).toBeInTheDocument(); await user.click(screen.getByRole("button",{name:"Next"})); expect(screen.getByText("second")).toBeInTheDocument(); await user.click(screen.getByRole("button",{name:"Previous"})); expect(screen.getByText("first")).toBeInTheDocument(); }); });
