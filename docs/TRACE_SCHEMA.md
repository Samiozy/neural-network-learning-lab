# Computation trace schema — version 1.0.0

Every simulation returns one JSON-safe document. Top-level identity and replay
fields are `trace_version`, `simulation_id`, `architecture`, `seed`,
`precision_policy`, and `config`. Parameter lineage is recorded in
`initial_parameters`, `gradients`, and `final_parameters`. `warnings` and `errors`
are explicit arrays; NaN and Infinity are rejected rather than serialized as null.

Each ordered `steps` entry contains `step_id`, `phase`, `operation`, `description`,
`equation_latex`, `substitution_latex`, `input_refs`, `output_refs`, `input_shapes`,
`output_shapes`, `values`, `tensor_indices`, `highlight_targets`,
`derivation_notes`, and `code_mapping`. Previous/next behavior uses this immutable
array directly. MVP tensors are deliberately bounded, so values are stored inline;
future large tensors must use versioned references or bounded snapshots.

The complete DNN fixture used for schema and replay tests is the 2→3→1 example in
`apps/api/tests/test_core.py`. Schema changes require a semantic version change,
fixture migration, and backward-compatibility note.

