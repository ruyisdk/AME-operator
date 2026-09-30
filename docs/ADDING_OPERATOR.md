# Adding an Operator

Keep each operator independent from compiler and simulator source trees.

1. Define the API, shapes/layout, workspace, ownership, numeric domain and error
   precedence in `operators/include/` and `operators/docs/`.
2. Implement scalar validation/packing under `operators/common/` and AME code
   under `operators/kernels/<operator>/`. Separate typed kernel translation
   units from ordinary pointer/scalar wrappers.
3. Write a logical GOLDEN in `operators/reference/` that does not reuse kernel
   packing or simulator outputs. Add literal checks and a second reference
   where rounding/overflow makes it necessary.
4. Add a versioned input catalog under `operators/cases/`, fixture generator and
   target driver under `operators/tests/`. Cover tails, multiple blocks, zero
   dimensions, errors, guards and state preservation.
5. Add an entry to `operators/registry.json`: unique operator/variant/case IDs,
   sources, typed/C++ sources, capabilities, profiles, contract, fixture inputs
   and expected instruction scans. Add a portable suite config.
6. Extend frame and result verification for the new kernel and exact assertion
   inventory. Test missing, reordered and corrupted results explicitly.
7. Run host tests, dual-XLEN build, smoke, full matrix and deliberate fault
   controls. Publish commands and selected results with the new variant.

Registry entries are data; `scripts/operators.py` enumerates profile, XLEN,
optimization and language axes. Execution adapters remain shared. Numerical
accounting and frame analysis have explicit per-variant dispatch so that a new
operator cannot silently inherit an unrelated validation rule.
