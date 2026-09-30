#include "../../common/pack_bf16.h"
#include "../../common/pack_f32.h"
#include <riscv_ztt.h>
void ame_ops_gemm_bf16_kernel (ame_size m, ame_size n, ame_size k,
                               const ame_bf16 *a, ame_size lda,
                               const ame_bf16 *b, ame_size ldb,
                               float *c, ame_size ldc, int accumulate,
                               ame_bf16 *ap, ame_bf16 *bp, float *cp, ame_size t)
{
  for (ame_size i=0; i<m; )
    {
      ame_size mr=m-i<t ? m-i : t;
      for (ame_size j=0; j<n; )
        {
          ame_size nr=n-j<t ? n-j : t;
          if (accumulate)
            ame_ops_pack_f32 (cp,t,c+i*ldc+j,ldc,mr,nr,AME_OPS_F32_SEED);
          else
            for (ame_size q=0; q<t*t; ++q) cp[q]=0.0f;
          __riscv_ztt_f32_rne_accx1_t acc=__riscv_ztt_mcopy_m2a_f32_rne_accx1
            (__riscv_ztt_mls_rm_f32_rne_1x1 (cp));
          for (ame_size p=0; p<k; )
            {
              ame_size kr=k-p<2*t ? k-p : 2*t;
              ame_ops_pack_bf16_a (ap,t,a+i*lda+p,lda,mr,kr);
              ame_ops_pack_bf16_b (bp,t,b+p*ldb+j,ldb,kr,nr);
              __riscv_ztt_bf16_rne_1x2_t av=__riscv_ztt_mls_rm_bf16_rne_1x2 (ap);
              __riscv_ztt_bf16_rne_2x1_t bv=__riscv_ztt_mls_rm_bf16_rne_2x1 (bp);
              acc=__riscv_ztt_mmulacc_2d_f32_rne_accx1 (acc,av,bv);
              p+=kr;
            }
          __riscv_ztt_mss_rm (cp,__riscv_ztt_mcopy_a2m_f32_rne_1x1 (acc));
          ame_ops_unpack_f32 (c+i*ldc+j,ldc,cp,t,mr,nr);
          j+=nr;
        }
      i+=mr;
    }
}
