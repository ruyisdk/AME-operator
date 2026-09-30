#ifndef AME_OPS_H
#define AME_OPS_H
typedef __INT32_TYPE__ ame_i32;
typedef __SIZE_TYPE__ ame_size;
typedef __UINTPTR_TYPE__ ame_uintptr;
enum ame_ops_status {
  AME_OPS_OK, AME_OPS_BAD_ARGUMENT, AME_OPS_NO_WORKSPACE,
  AME_OPS_NOT_OWNED, AME_OPS_UNSUPPORTED
};
#ifdef __cplusplus
extern "C" {
#endif
ame_size ame_ops_i32_workspace_bytes (ame_size tile);
int ame_gemm_i32_owned (ame_size m, ame_size n, ame_size k,
                       const ame_i32 *a, ame_size lda,
                       const ame_i32 *b, ame_size ldb,
                       ame_i32 *c, ame_size ldc, int accumulate,
                       void *workspace, ame_size workspace_bytes);
int ame_gemv_i32_owned (ame_size m, ame_size k,
                       const ame_i32 *a, ame_size lda,
                       const ame_i32 *x, ame_i32 *y, int accumulate,
                       void *workspace, ame_size workspace_bytes);
#ifdef __cplusplus
}
#endif
#endif
