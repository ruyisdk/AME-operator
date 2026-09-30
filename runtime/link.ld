OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { ram (rwx) : ORIGIN = 0x80000000, LENGTH = 16M }
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
  __stack_bottom = .;
  .stack (NOLOAD) : { . += 2M; } > ram
  __stack_top = .;
  ASSERT(__stack_top <= ORIGIN(ram) + LENGTH(ram), "AME test RAM overflow")
}
