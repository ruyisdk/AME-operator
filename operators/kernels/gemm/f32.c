#include "../../common/pack_f32.h"
#include <riscv_ztt.h>

void ame_ops_gemm_f32_kernel (ame_size m, ame_size n, ame_size k,
                             const float *a, ame_size lda, const float *b, ame_size ldb,
                             float *c, ame_size ldc, int accumulate,
                             float *workspace, ame_size t)
{
  float *ap = workspace, *bp = ap+t*t, *cp = bp+t*t;
  for (ame_size i = 0; i < m; )
    {
      ame_size mr = m-i < t ? m-i : t;
      for (ame_size j = 0; j < n; )
        {
          ame_size nr = n-j < t ? n-j : t;
          if (accumulate)
            ame_ops_pack_f32 (cp,t,c+i*ldc+j,ldc,mr,nr,AME_OPS_F32_SEED);
          else
            for (ame_size q = 0; q < t*t; ++q) cp[q] = 0.0f;
          __riscv_ztt_f32_rne_accx1_t acc = __riscv_ztt_mcopy_m2a_f32_rne_accx1
            (__riscv_ztt_mls_rm_f32_rne_1x1 (cp));
          for (ame_size p = 0; p < k; )
            {
              ame_size kr = k-p < t ? k-p : t;
              ame_ops_pack_f32 (ap,t,a+i*lda+p,lda,mr,kr,AME_OPS_F32_LEFT);
              ame_ops_pack_f32 (bp,t,b+p*ldb+j,ldb,kr,nr,AME_OPS_F32_RIGHT);
              __riscv_ztt_f32_rne_1x1_t av = __riscv_ztt_mls_rm_f32_rne_1x1 (ap);
              __riscv_ztt_f32_rne_1x1_t bv = __riscv_ztt_mls_rm_f32_rne_1x1 (bp);
              acc = __riscv_ztt_mmulacc_2d_f32_rne_accx1 (acc,av,bv);
              p += kr;
            }
          __riscv_ztt_mss_rm (cp,__riscv_ztt_mcopy_a2m_f32_rne_1x1 (acc));
          ame_ops_unpack_f32 (c+i*ldc+j,ldc,cp,t,mr,nr);
          j += nr;
        }
      i += mr;
    }
}
