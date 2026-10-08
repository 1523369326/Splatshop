# GaussianSelector 实际验证记录

## 2026-10-08：统一主构建与独立 Dev 入口

- 已将主 `build` 缓存切换为 selector ON / fetch deps ON，复用已缓存的 igraph 源码到 `build/_deps/igraph-src`；不依赖历史独立构建目录运行。
- `cmake --build build --config Release --target SplatEditor gaussian_selector_tests -- /m` 成功，exit 0。实际产物 `build/Release/SplatEditor.exe`，2,699,776 bytes。仍有现有 igraph 字符集、GLEW 链接声明及 OpenVR 等警告，未声称零警告构建。
- `ctest --test-dir build -C Release --output-on-failure --timeout 30`：1/1 passed，0 failed。
- `git diff --check` 通过。源码核对：Dev 独立按钮位于 Dev&Debug 之后，调用独立窗口；action Start/Controls 指向新窗口；原 Appearance 中 Sort-Free/Core-Tail/RaDe 与其 Controls 保留；删除重复 GS 工具栏入口。
- 未进行本轮 GUI 点击验收；算法和 CUDA coverage 未修改，本轮没有重复 GPU 数值测试。旧独立构建目录保留为历史产物，下方旧日期命令仅是历史记录。

## 2026-10-01：作者笔记配置对齐

- 默认T=.9、16owners、EM<=80、adaptive graph、mixed contrast、两阶段SP refinement、整数cut及postfilters的Feature ON Release构建成功（exit0）。最后主程序：`build/gaussian-selector-release/Release/SplatEditor.exe`。
- 核心测试1/1通过，0failed，约.02s。新增随机N=1..9量化能量穷举核对native push-relabel最优性、极端capacity动态缩放及非法输入；GMM min(3,n-1)/singleton kNN；Q10/soft votes/support-center gate；adaptive voxel budget、不丢成员、below-budget exact、deterministic mapping；4个真实uncertain SP拆分及member cap；component/scale过滤开关；最多32个pseudoBG声明。
- `python tests/gaussian_selector/coverage_gpu_check.py` exit0：实际CUDA coverage body fixture验证owner cap、support、2%relative center-visible tolerance，加上既有跨batch、T、FG/BG、完整视图分母、opaque depth、ROI外遮挡与空tile；真实renderer完整NVRTC编译通过。简化packing/decoder，不等于真实场景数值对拍。
- `git diff --check`通过。未执行用户真实模型GUI质量测试；公开参数与E补全的区别见parameters.md/source_alignment.md。
- Feature OFF核心构建（`build/gaussian-selector-tests`）成功，ctest同样1/1通过；缺失igraph时仍明确拒绝Leiden，没有fallback聚类。

## 本轮作者笔记优化：实际通过

- Feature ON Release：`cmake --build build/gaussian-selector-release --config Release --target gaussian_selector_tests SplatEditor -- /m` 构建成功，exit code 0。包括前轮尚未编译的 ROI 快照预算修改及本轮 coverage、conflict refinement、UI、共享编码与诊断修改。
- `ctest --test-dir build/gaussian-selector-release -C Release --output-on-failure`：1/1 passed，0 failed，约 0.02s。新增回归覆盖：T=.5 遮挡后层、FG/BG 对称、完整视图分母、invalid-row 外层遮挡、pre-GS opaque depth、跨 batch 终止、depthwrite-disabled、冲突 SP 拆分、原 Gaussian row 映射、重复 solve 与 base cache 不变、split budget 明确拒绝。
- `python tests/gaussian_selector/coverage_gpu_check.py`：成功，exit code 0。实际执行 production coverage `.cuh` 的 GPU 合成 fixture，2 个 staging batches，覆盖标准/T=.5/T=.9、FG/BG、unpainted denominator、opaque depth、ROI 外遮挡、disabled depthwrite、空 tile；可解析质量与 CPU 计算误差小于 1e-6。fixture 简化 StageData packing/decoder，不能视为真实 renderer 的完整 ABI/数值对拍。
- 同一 GPU 脚本用生产 NVRTC flags、真实类型成功编译完整 `src/gaussians_rendering.cu`（selector ON），未打开编辑器窗口；不是 GUI Apply 的人工验收。
- `git diff --check` 通过。
- 构建环境最初存在 PATH/Path 重复导致 MSBuild 检测失败，留下失败的 sinh 缓存；用干净 subprocess environment 和 `cmake -U SINH_FUNCTION_EXISTS` 重新配置后通过。未修改依赖源码或伪造数学库检测成功。

仍需使用用户真实模型比较相同 ROI/笔画下标准 replay、T=.5、T=.9 的 Preview，并验证 Apply/Undo 和薄结构漏选。自动测试不代表已经解决所有后方 Unknown Gaussian 的误分类。

## 前轮 ROI 快照预算修复（当时仅源码检查；本轮已构建）

针对 5,834,784 个 SH3 Gaussian 在 Apply 时出现 `selector CPU budget exceeded`：快照改为 16,384 行一块扫描 position/flags，先按屏幕 ROI 筛选，再仅对候选连续索引段读回 scale/quaternion/color/SH。取消整层完整属性快照的预检，改为 identity map、固定传输 scratch、候选快照和编码工作空间的预算预检。ROI/相机变化会重新捕获快照，防止扩大 ROI 后复用旧范围。框外 identity 为 invalid，但仍参与 raster 的透射率累计；Apply 的域由原始 localIndex 恢复。

本次仅完成源码调用点、索引映射及 `git diff --check` 检查，按用户要求未执行编译/运行。下面构建与测试记录属于之前版本。建议用同一模型分别测试小 ROI、扩大/移动 ROI、重复 Apply 与 Undo；真实候选超限仍会明确报 `ROI candidate count exceeds ...`，需要缩小框。

环境：Windows 11，Visual Studio 2022/MSBuild 17.14，CUDA Toolkit 12.8，NVRTC `compute_89/sm_89`。日期：2026-09-29。

## 构建与启动结果

1. Feature ON Release 主程序：

   ```powershell
   cmake --build build/gaussian-selector-release --config Release --target SplatEditor -- /m
   ```

   结果：成功，exit code 0。链接到固定的 igraph 1.0.0；仅有仓库既有 MSVC/GLEW warnings。

2. Core tests：

   ```powershell
   cmake --build build/gaussian-selector-release --config Release --target gaussian_selector_tests -- /m
   ctest --test-dir build/gaussian-selector-release -C Release --output-on-failure
   ```

   结果：1/1 passed，0 failed；最终复测总测试时间约 0.02 s。

3. Feature OFF Release：首次新配置因受限网络无法 clone GLFW 而失败；随后显式复用工作区已有 `build/_deps/glfw-src`，离线配置和构建成功，exit code 0：

   ```powershell
   cmake -S . -B build/gaussian-selector-off `
     -DSPLATSHOP_ENABLE_GAUSSIAN_SELECTOR=OFF `
     -DSPLATSHOP_BUILD_GAUSSIAN_SELECTOR_TESTS=OFF `
     -DFETCHCONTENT_SOURCE_DIR_GLFW=E:/CloudPointTest/3dgs_view/Splatshop/build/_deps/glfw-src
   cmake --build build/gaussian-selector-off --config Release --target SplatEditor -- /m
   ```

4. Feature ON NVRTC/启动：隐藏启动 25 s。首次测试发现 selector 宏只传给 renderer，造成其它 CUDA 模块与 host 的 `GaussianData` ABI 不一致，并在 `kernel_compute_mortoncodes_32bit` 触发 CUDA 700 illegal address。修复为 ON 时所有 CUDA module 统一定义 selector 宏、禁用旧 cubin 后复测：

   - `kernel_gaussian_selector_capture_depth` 与 `kernel_gaussian_selector_coverage` 均成功编译/link；
   - 最终变更后所有 CUDA programs 在约 8.9 s 完成；
   - 继续运行内置 brush sphere 的 sort/render，没有再出现 CUDA error/illegal address；
   - 日志仅保留缺少 `./splatmodels/scene.json` 的既有启动提示；最终日志为 `build/gaussian-selector-release/selector-runtime-coverage-final.out.log`。

## 已覆盖的自动测试

- CAC：SH0、单一 SH1 signed direction、q/-q。
- graph：重复坐标、有限权重、canonical undirected dedupe。
- Leiden：真实 igraph weighted smoke。
- min-cut：1/2 节点 terminal 方向；随机 N=1..10 与独立全枚举能量一致。
- coverage CPU oracle：前后层 T、完整视图 denominator。
- evidence：跨视图 FG/BG 平票为 Unknown。
- GMM：seed median 约映射到 0.95/0.05。
- selection：Replace/Undo/Redo 精确恢复，域外及其它 flag bit 不变。
- experimental refinement：显式模式、1–3 次上限、工程变体诊断、保留 reservoir 的执行路径。

## 尚未完成的验证

- 尚未使用用户实际 Gaussian 场景人工操作 GS toolbar 完成一次 Solve/Preview/Apply/Undo 全流程。
- coverage 尚无独立 GPU/CPU 数值对拍：跨 tile、多个 staging batch、完全 opaque、其它 node occluder。
- 自动 ROI 的扩大恢复、cache cold/hit 等价、old/new request 与长期内存尚无自动压力测试。
- 没有 active solve 的 median/p95、峰值 CPU/GPU 内存或真实质量截图。
