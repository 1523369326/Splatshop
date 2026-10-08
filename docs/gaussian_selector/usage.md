# GaussianSelector 使用说明

## 构建

```powershell
cmake -S . -B build `
  -DSPLATSHOP_ENABLE_GAUSSIAN_SELECTOR=ON `
  -DSPLATSHOP_BUILD_GAUSSIAN_SELECTOR_TESTS=ON `
  -DSPLATSHOP_FETCH_GAUSSIAN_SELECTOR_DEPS=ON
cmake --build build --config Release --target SplatEditor gaussian_selector_tests -- /m
ctest --test-dir build -C Release --output-on-failure
.\build\Release\SplatEditor.exe
```

## 编辑器操作

### ROI 交互入口（更新）

ROI 快照先分块扫描位置与 flags，再读取框内候选的完整属性/SH；不会仅因为整层 SH 超过 1 GiB 而拒绝小 ROI。状态栏 `Scope` 显示实际候选数 / 原始图层总数。候选上限按 CPU 快照预算计算；图默认最多50k节点，超出后 adaptive voxel pooling，不丢弃 Gaussian 成员。

在顶部 `Dev&Debug` 旁点击独立的 `Dev` 按钮，展开 `GaussianSelector`，点击 `Start GaussianSelector`。
`Dev` 是独立窗口，目前只包含 GaussianSelector；未来新增功能可在 `src/gui/devFeatures.h` 增加节点。Sort-Free、Core-Tail、RaDe-GS 保留原有 Appearance/Controls 入口。Feature OFF 构建仍显示 GaussianSelector 的启用说明。

主构建默认启用 GaussianSelector，直接集成到同一个 SplatEditor，不需要第二个编辑器。上面的显式 ON 参数也用于更新旧 build 中保存的 OFF 缓存。第一次构建需要下载固定版本 igraph，或提供本地 `FETCHCONTENT_SOURCE_DIR_IGRAPH`。旧 `build/gaussian-selector-release` 是历史构建目录，不再作为运行入口。请从仓库根目录启动，以便找到运行时 CUDA 源码。

先在视口拖动黄色 ROI 矩形，再使用 Foreground / Background / Erase 在框内绘制。重新画框会清掉旧笔画。Preview 只显示结果；Apply selection 会自动求解并将结果写入普通选区，支持移动、复制、删除和导出。Apply 后 Ctrl+Z/Y 操作模型选区历史，Undo/Redo stroke 按钮仍可显式操作笔画。

屏幕 ROI 按当前视图中 Gaussian 中心投影定义求解范围，属于工程交互实现，不声明为作者默认值。框外仍参与遮挡和完整视图 coverage 分母；四种 Apply 操作均保留 ROI 外已有选择。切换相机需要在新视图重新画 ROI；当前矩形工作流切换 scope 时清除旧 votes。边界处中心在框外但椭圆与框相交的 Gaussian 不在求解范围内。

推荐先点击 `Use author-notes profile`：T=.9、16 owners、EM<=80、k=8 weighted contrast、两阶段局部 SP refinement、integer cut。按钮保留你的 .35/.40/.25 权重与 CPU/GPU 预算。先用真实 FG/BG 笔画测试，再尝试无 BG 的弱 pseudo-BG，状态会明确显示使用数量。

主要控制为 node budget、foreground-seeded components、oversized rejection；后两者的具体过滤规则是 E，漏选分离花瓣/大尺度 Gaussian 时可关闭对照。Algorithm parameters 提供混合模型、post-cut refinement、integer cut、center gate、T presets 等诊断开关。Standard replay 按钮同时取消 T 与 owner budget，便于比较。

选择一个可见、未锁定的 Gaussian layer，并使用标准 single-view 3DGS raster kernel；关闭 SoA/bandwidth/fragment、RaDe、Heatmap/Core-Tail、Solid、dual culling 与 multi-view 变体。绿色 overlay 是独立 preview，不改变 `FLAGS_SELECTED`。Cancel 或切换工具不会提交选择。原工具栏的重复 GS 入口已移除，统一从 Dev 启动。

默认从固定 immutable base 开始，在请求内做作者笔记的两阶段 SP split。`Experimental FG refinement` 另加至多2次 foreground-restricted 工程循环，需要真实 BG，不是作者笔记 split 的同义开关；测试笔记配置时保持关闭。

无显式框的旧接口超过500k时先 lifting，再生成自动3D ROI；当前屏幕矩形流程直接限定 snapshot 域。超 CPU/GPU 预算时明确拒绝，不会随机抽样。完整 raster 与 ROI 外遮挡保持不变。
