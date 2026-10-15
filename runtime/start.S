/* Entered from spike_boot.S: one M-mode hart, MS enabled, IRQs disabled. */
  .section .text.init,"ax",@progbits
  .global _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, tutorial_trap
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sw zero, 0(t0)
  addi t0, t0, 4
  j 1b
2:
  call tutorial_main
  j tutorial_done

  .balign 4
tutorial_trap:
  la t0, tutorial_mailbox
  li t1, 0x414d4534
  sw t1, 0(t0)
  li t1, 1
  sw t1, 4(t0)
  li t1, 4
  sw t1, 12(t0)
  csrr t1, mcause
  sw t1, 48(t0)
  srli t1, t1, 32
  sw t1, 52(t0)
  csrr t1, mepc
  sw t1, 56(t0)
  srli t1, t1, 32
  sw t1, 60(t0)
  csrr t1, mtval
  sw t1, 64(t0)
  srli t1, t1, 32
  sw t1, 68(t0)
  li t1, 3
  sw t1, 8(t0)
  j tutorial_done

  .global tutorial_done
  .type tutorial_done,@function
tutorial_done:
  fence rw,rw
  .global tutorial_stop
tutorial_stop:
3:
  j 3b
  .size tutorial_done,.-tutorial_done
