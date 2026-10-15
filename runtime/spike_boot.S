/* Pinned developer Spike uses provisional AME MS bits 26:25.
   This model-specific adapter is separate from the tutorial ELF. */
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
