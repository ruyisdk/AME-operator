/* Entry adapter for Spike ead03459: AME MS uses provisional bits 26:25.
   Loaded as a separate payload; the test ELF is never rewritten. */
  .section .text,"ax",@progbits
  .option norvc
  .global spike_boot
spike_boot:
  csrw mie, zero
  li t0, 8
  csrc mstatus, t0
  li t0, 0x02000000
  csrs mstatus, t0
  li t0, 0x80000000
  jr t0
