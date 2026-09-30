# Platform Requirements

## Compiler

Use AME/Ztt-enabled GCC and Binutils with ISA v0.6 instructions and the
intrinsic v0.2.5 interface. Kernel compilation uses:

```text
-march=rv32im_zicsr_ztt0p6 -mabi=ilp32
-march=rv64im_zicsr_ztt0p6 -mabi=lp64
-mztt-profile=gcc-runtime-u32-m16-a4
```

The source and build-tree revisions used for this export are recorded in
[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md). Standard upstream compilers
without AME support cannot build these kernels.

## CSR and Resources

| CSR | Address |
| --- | --- |
| amenlen | 0xCC0 |
| ameudsz | 0xCC1 |
| ameown | 0xCC3 |
| amestatus | 0x800 |
| amestype | 0x810 |
| amefflags | 0x811 |
| amexsat | 0x812 |

The test platform supplies UDS32, 16 M registers, 4 Acc registers and runtime
tile side T in {4,8,16,32,64,128}. The model's `--ame-nelem` is T*T. Matrix
problem dimensions m/n/k are independent of T. The unsupported-platform suite
additionally requests UDS8/16/64 and T2/T256 to test rejection.

## Spike Adapter

The AME developer Spike must implement Ztt, AME resource CLI options,
`ame 0` debug observations and raw memory reads. The adapter loads the test
at 0x80000000 and bootstrap at 0x80F00000 in a 16-MiB RAM region. Entry is
M-mode, one hart, interrupts disabled, MS enabled and AME resources unowned.
State/FP suites enable standard F to observe FCSR; kernels keep the ISA flags
shown above. The bootstrap and target startup are in `runtime/`.

The selected model base is `ead03459a3d9db96788d2d2fe626d0b9f3bcd367`, with CSR,
ownership, wide-scalar, store, arithmetic and FP fixes. The export validation
uses the combined fixes plus the **BF16 NV r2** correction. It preserves the
destination rounding mode while accumulating BF16 signaling-NaN conversion NV.
The earlier BF16 NV r1 fix changed RNO behavior; use r2 as its replacement.
Obtain the matching model and fixes from the AME Spike developers; simulator
source and binaries are separate dependencies of this repository.

Required numerical behavior for the FP RNE suites: increasing-K fused FP32
accumulation, canonical NaN, signaling-NaN NV, after-rounding tininess and sticky
flags. Backend changes should be recorded with a new model revision.

## Other Execution Backends

An adapter receives `--request <json> --response <json>`. The request includes
ELF/hash, symbols, nonce, model revision and expected hardware configuration.
The response reports matching identity plus observed configuration, completion
PC and a 128-byte mailbox as hex. `scripts/ame.py` validates the response.
Configure an absolute adapter command and explicit capabilities in a platform
JSON, then use `scripts/operators.py check --mode execute`. A SAIL adapter can
reuse this protocol and the same linked ELFs and GOLDEN.
