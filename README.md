# RISC-V AME GCC Tutorial

以 **Jupyter Notebook** 学习 C/C++ INT32 GEMM：编译、反汇编、Spike 执行及独立参考校验。
主配置为 **RV64、UDS32/M16/Acc4、N=4**。

打开 [AME_Tutorial.ipynb](AME_Tutorial.ipynb)，选择 Python 3 内核，从上到下运行。
Notebook 包含中文讲解、两张示意图、C/C++ 示例、可执行练习及源码重建说明。
可使用 VS Code 的 Jupyter 支持或已安装的 JupyterLab。

尚未安装 Jupyter 时，可在项目内创建独立环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install jupyterlab
.venv/bin/jupyter lab AME_Tutorial.ipynb
```

## 快速运行

本机目录已有配套工具，可直接执行：

```bash
cd ~/ame-tutorial-gcc
python3 scripts/check-host.py
bash scripts/present.sh
```

完整验证 C/C++ × O0/O2 与错误注入：

```bash
bash scripts/acceptance.sh
```

生成的 ELF、反汇编和数值结果写入 Git 忽略的 `build/`。
脚本不生成额外日志文件或 SHA 校验清单，编译与执行错误直接显示在终端。

## 仓库内容

```text
AME_Tutorial.ipynb       主教程与可执行练习
examples/               C/C++ GEMM 与输入数据
runtime/                裸机启动、链接脚本和结果邮箱
scripts/                运行、数值校验和重建脚本
docs/images/           两张教学图
docs/DEMO_2_3_MIN.md    150 秒讲者稿
VERSION.json            配套源码版本与目标配置
VALIDATION.md           简短验证结果
licenses/               上游许可
```

`toolchain/`、`spike/` 和 `sources/` 是本机依赖，不进入 Git。
**仅克隆此仓库还不能运行 AME 示例**：需另外放入匹配的工具安装，
或取得对应的完整源码后按 Notebook 重建。发布时可将配套工具和源码作为独立附件提供；
本仓库尚未设置下载地址。

## 环境与范围

- 宿主：x86_64 Linux、Python 3.11+、`dtc` 及工具所需系统动态库。
- 目标：freestanding C/C++，包含 libgcc；不包含 newlib、libstdc++ 或操作系统。
- 已实测：Ubuntu 24.04.2 LTS / glibc 2.39。
- 待验收：Ubuntu 26.04.1 LTS / glibc 2.43，可用 `bash scripts/acceptance.sh --require-target` 检查。

教学例子只覆盖无溢出的 INT32 NN GEMM；Spike 结果用于功能校验，不代表硬件性能。
