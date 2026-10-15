OUTPUT_ARCH(riscv)
ENTRY(spike_boot)
PHDRS { boot PT_LOAD FLAGS(5); }
SECTIONS {
  . = 0x80f00000;
  .text : { *(.text) } :boot
  /DISCARD/ : { *(.riscv.attributes) }
}
