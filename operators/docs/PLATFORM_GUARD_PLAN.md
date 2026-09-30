# Unsupported-Platform Validation

Production checks ownership, UDS==32 and power-of-two T in [4,128] before
entering typed kernels. A dedicated scalar harness records actual hardware
resources separately from the compiled UDS32/M16/Acc4 profile.

| Configuration | Expected compute result |
| --- | --- |
| UDS32, T4 or T128 | OK |
| UDS8/T4, UDS16/T8, UDS64/T4 | UNSUPPORTED |
| UDS32, T2 or T256 | UNSUPPORTED |

All cases use M16/Acc4 and actual model configurations. Each ELF has 100
scenarios across five production paths. Both XLENs, C/C++ and O0/O2 are covered.
Released calls return NOT_OWNED before workspace/descriptor access. Invalid
arguments, empty output and K=0 retain their ordinary precedence.

Tests compare complete small inputs/output/work buffers, descriptor fields,
ownership, UN, flags/xsat and FCSR. Positive computations use separate native
large work arrays whose immediate guard bands are checked. Broader workspace
coverage belongs to the per-family state suites.

Selected O2/C runs trace bounded driver regions. Rejected configurations have
no typed-kernel entry; T4 controls have four entries per family. Trace logs are
losslessly compressed with raw/compressed hashes and a 256-MiB limit.

Commands are in [BUILD_AND_RUN](../../docs/BUILD_AND_RUN.md). The verifier
requires 16 builds, 32 probes, 112 main executions, 32 traced executions and
cross-path agreement. Normal numerical executors retain strict profile checks.
