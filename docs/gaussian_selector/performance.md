# GaussianSelector 性能记录

当前只能报告构建、核心 fixture 与启动编译数据，尚不能报告真实场景交互求解性能。

- Feature OFF Release 增量构建：成功；一次记录约 25 s（受当次重编译文件影响），另一次无改动构建约 1.25 s。
- Feature ON 完整链接（igraph 已构建缓存）：约 6.6 s。
- CPU core tests：CTest 实际总时约 0.02 s；fixture 很小，不能外推到场景。
- 修复 Host/Device ABI 后，Feature ON 的全部 CUDA programs 使用一致 selector struct 宏；最终变更后的运行时编译和链接约 8.9 s，coverage/capture kernel 只定义在 `gaussians_rendering.cu`。统一宏是结构 ABI 要求，不代表其它 kernel 执行 selector 工作。

未量测：cold encoding、Gaussian kNN、Leiden、lifting/readback、GMM、cut、preview upload、end-to-end median/p95、idle overhead、峰值 CPU/GPU 内存。宿主链路已经接通，但当前工作区没有用于完成真实交互质量/性能矩阵的用户场景与固定相机轨迹。
