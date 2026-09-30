/* Adapter enters M-mode on one hart, with all applicable MS fields enabled.
   No guessed privileged MS bit or simulator-specific exit syscall is used. */
  .section .text.init,"ax",@progbits
  .global _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, ame_trap
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sw zero, 0(t0)
  addi t0, t0, 4
  j 1b
2:
  call test_main
  j ame_test_done

  .balign 4
ame_trap:
  la t0, ame_mailbox
  li t1, 0x414d4554
  sw t1, 0(t0)
  li t1, 1
  sw t1, 4(t0)
  li t1, 6
  sw t1, 44(t0)
  csrr t1, mcause
  sw t1, 64(t0)
#if __riscv_xlen == 64
  srli t1, t1, 32
  sw t1, 76(t0)
#endif
  csrr t1, mepc
  sw t1, 68(t0)
#if __riscv_xlen == 64
  srli t1, t1, 32
  sw t1, 80(t0)
#endif
  csrr t1, mtval
  sw t1, 72(t0)
#if __riscv_xlen == 64
  srli t1, t1, 32
  sw t1, 84(t0)
#endif
  li t1, 3
  sw t1, 8(t0)
  j ame_test_done

  .global ame_test_done
  .type ame_test_done,@function
ame_test_done:
  fence rw,rw
3:
  j 3b
  .size ame_test_done, .-ame_test_done
