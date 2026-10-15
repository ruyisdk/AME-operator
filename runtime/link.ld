OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { ram (rwx) : ORIGIN = 0x80000000, LENGTH = 15M }
SECTIONS {
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > ram
  .rodata : { *(.rodata .rodata.*) } > ram
  .data : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.* .data .data.*)
  } > ram
  .bss (NOLOAD) : {
    . = ALIGN(16);
    __bss_start = .;
    *(.sbss .sbss.* .bss .bss.* COMMON)
    . = ALIGN(16);
    __bss_end = .;
  } > ram
  . = ALIGN(16);
  .stack (NOLOAD) : { . += 64K; } > ram
  __stack_top = .;
  ASSERT(__stack_top <= 0x80f00000, "ELF overlaps Spike bootstrap")
}
