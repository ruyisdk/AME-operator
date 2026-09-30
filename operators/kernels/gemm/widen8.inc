#include "../../common/pack_i8.h"
#include "../../common/pack_i32.h"
#include <riscv_ztt.h>

void WIDEN_KERNEL (ame_size m, ame_size n, ame_size k,
                   const void *a0, ame_size lda, const void *b0, ame_size ldb,
                   ame_i32 *c, ame_size ldc, int accumulate, void *workspace, ame_size t)
{
  const ame_u8 *a = (const ame_u8 *) a0, *b = (const ame_u8 *) b0;
  ame_u8 *ap = (ame_u8 *) workspace, *bp = ap+4*t*t;
  ame_i32 *cp = (ame_i32 *) (bp+4*t*t);
  for (ame_size i = 0; i < m; )
    {
      ame_size mr = m-i < t ? m-i : t;
      for (ame_size j = 0; j < n; )
        {
          ame_size nr = n-j < t ? n-j : t;
          if (accumulate) ame_ops_pack_i32 (cp,t,c+i*ldc+j,ldc,mr,nr);
          else
            for (ame_size q = 0; q < t*t; ++q) cp[q] = 0;
          __riscv_ztt_i32_rnu_accx1_t acc = __riscv_ztt_mcopy_m2a_i32_rnu_accx1
            (__riscv_ztt_mls_rm_i32_rnu_1x1 (cp));
          for (ame_size p = 0; p < k; )
            {
              ame_size kr = k-p < 4*t ? k-p : 4*t;
              ame_ops_pack_i8_a (ap,t,a+i*lda+p,lda,mr,kr);
              ame_ops_pack_i8_b (bp,t,b+p*ldb+j,ldb,kr,nr);
              WIDEN_ROW av = WIDEN_LOAD_ROW ((const WIDEN_ELEMENT *) ap);
              WIDEN_COL bv = WIDEN_LOAD_COL ((const WIDEN_ELEMENT *) bp);
              acc = __riscv_ztt_mmulacc_2d_i32_rnu_accx1 (acc,av,bv);
              p += kr;
            }
          __riscv_ztt_mss_rm (cp,__riscv_ztt_mcopy_a2m_i32_rnu_1x1 (acc));
          ame_ops_unpack_i32 (c+i*ldc+j,ldc,cp,t,mr,nr);
          j += nr;
        }
      i += mr;
    }
}
