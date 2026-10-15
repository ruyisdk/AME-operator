# 验证结果

验证环境：x86_64、Ubuntu 24.04.2 LTS、glibc 2.39。
教学配置：RV64、UDS32/M16/Acc4、N=4、INT32 NN GEMM。

- GCC、Binutils 与修复后的 Spike 从独立源码构建完成。
- C/C++ × O0/O2 四个组合完成编译、反汇编、Spike 执行及独立 Python 参考比较。
- 全部 16 个输出、8 个边界哨兵、完成 PC、AME 配置与 ownership 检查通过。
- 错误注入准确检出 `C[0][0]` 从 -10 变为 -9；超时或陷阱不算负控成功。
- 单位矩阵练习、缺少 ISA/profile 的诊断均已实际验证。
- 主教程符合 nbformat 4.5，14 个 Python 代码单元已全部执行通过；仓库文件不保存运行输出。
- 工具链搬迁验证与 Spike 五个定向回归通过。

原始构建日志和历史运行目录不纳入此教学项目。可随时重跑：

```bash
bash scripts/acceptance.sh
```

输出应为四个 `PASS`、一个 `EXPECTED_REJECTION`，最后显示 `ACCEPTANCE_PASS`。
数值参考保持独立；本项目不保留额外文件 hash 校验流程。

Ubuntu 26.04.1 / glibc 2.43 尚未实测。没有硬件或 SAIL 验收，也不据此报告加速比。
