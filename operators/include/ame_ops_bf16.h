#ifndef AME_OPS_BF16_H
#define AME_OPS_BF16_H
#include "ame_ops_f32.h"
typedef __bf16 ame_bf16;
typedef char ame_bf16_storage_size[sizeof (ame_bf16) == 2 ? 1 : -1];
/* Three live object arrays; sizes count elements, not bytes.  */
struct ame_ops_bf16_workspace {
  ame_bf16 *a;
  ame_size a_elements;
  ame_bf16 *b;
  ame_size b_elements;
  float *c;
  ame_size c_elements;
};
#ifdef __cplusplus
extern "C" {
#endif
ame_size ame_ops_bf16_input_elements (ame_size tile);
ame_size ame_ops_bf16_output_elements (ame_size tile);
int ame_gemm_bf16_f32_owned (ame_size m, ame_size n, ame_size k,
                            const ame_bf16 *a, ame_size lda,
                            const ame_bf16 *b, ame_size ldb,
                            float *c, ame_size ldc, int accumulate,
                            const struct ame_ops_bf16_workspace *workspace);
int ame_gemv_bf16_f32_owned (ame_size m, ame_size k,
                            const ame_bf16 *a, ame_size lda,
                            const ame_bf16 *x, float *y, int accumulate,
                            const struct ame_ops_bf16_workspace *workspace);
#ifdef __cplusplus
}
#endif
#endif
