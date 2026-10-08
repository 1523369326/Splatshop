# GaussianSelector 集成图与阶段状态

集成入口核对日期：2026-10-08。下方 P1–P7 表保留早期阶段记录；算法后续优化及验证见 author_notes_optimization.md 与 validation.md，不把未做过的真实场景质量验证写成“已通过”。

## 改动范围审计（2026-10-08）

当前工作区的 Git diff 混有多视图、运行环境适配等改动，不能把所有 modified 文件算作 GaussianSelector 的成本，也不能据此整体回滚。按代码职责审计如下：

| 文件/范围 | 是否为选择器所需 | 原因/本轮处理 |
|---|---|---|
| `src/gaussian_selector/*` | 算法模块必需；拆成多少文件是组织选择 | CAC、图、Leiden、证据、建模、cut、细化独立，合成一个文件不会减少功能或耦合 |
| `SceneSnapshotAdapter.*`、`GaussianSelectorRuntime.*` | 必需 | 真实数据快照、身份映回、GPU 资源、异步请求、缓存、Preview/Apply/Undo |
| `HostDeviceInterface.h`、`gaussians_rendering.cu`、`SplatEditor_draw.h` 中 selector 接点 | 必需 | 真实 stage 身份、pre-GS depth、coverage 与独立预览；这些文件中的多视图修改另属其它功能 |
| `include/CudaModularProgram.h` 的 compileOptions、`SplatEditor.cpp` 的 NVRTC 宏传递 | 必需 | CPU 与所有 CUDA 模块必须使用同一结构布局，避免 ON/OFF ABI 不一致 |
| `scene/SNSplats.h`、`SplatEditor.cpp` 的 epoch/revision | 必需 | 排序/合并/变换/颜色/删除后使旧快照失效，防止异步结果落到错误 Gaussian |
| `BrushSelectAction.h`、`SpherePaintAction.h`、三个 `VR_*Action.h` 的版本通知 | 当前架构下必需 | 删除、涂色、变换及其 Undo/Redo 也会使缓存失效；未来统一 mutation API 后可集中，但直接删除通知会失去保护 |
| `InputAction.h`、`update/inputHandlingDesktop.h` | 交互接线必需 | 笔画历史与模型选区历史分流 |
| `SplatEditor.h/.cpp`、`gui/menubar.h`、`gui/devFeatures.h`、`GaussianSelectorAction.h` | UI 接线必需 | 独立 Dev 按钮、窗口及 action；本轮移除旧 Dev&Debug 内嵌入口 |
| `gui/toolbar.h` 的重复 GS 按钮；Dev 中其它渲染器的重复按钮 | 非必需 | 本轮移除；Sort-Free/Core-Tail/RaDe 原有 Appearance/Controls 入口保留 |
| `SplatEditor_render.h`、`SplatsManagement.h` 的四视图/Cov3D 改动 | 非选择器所需 | 工作区其它功能，本轮保留，未作为选择器改动撤回 |
| `cmake/common.cmake` 与 CUDA toolkit 路径/DLL 处理 | 运行环境适配 | 非选择算法本身；保留避免破坏现有启动环境 |
| `CMakeLists.txt`、`cmake/gaussian_selector.cmake` | 构建接线必需 | 静态链接到同一个 SplatEditor；默认启用，只使用主 build 目录 |
| `tests/gaussian_selector/*`、文档、原始 PromptPack/zip | 开发/验证资料 | 不属于编辑器运行时；不要求用单独编辑器 exe 承载 |

规范运行入口为 `build/Release/SplatEditor.exe`。历史 `build/gaussian-selector-release` 不再用于当前构建或操作说明；其文件暂留作旧产物，未删除用户文件。`gaussian_selector_tests.exe` 是回归测试程序，不是第二个编辑器。

## 已接通的主体链路

```text
SceneSnapshotAdapter（node/localIndex/indexEpoch/contentRevision/transform hash）
  -> CAC18（6 个 Gaussian 主轴方向的宿主 SH 语义）
  -> Gaussian 位置 kNN + weighted Leiden
  -> mean-pooled superpoints + SP kNN graph
  -> FG/BG/erase 多视图 stroke masks
  -> pre-GS opaque depth capture
  -> standard stage/tile alpha-transmittance coverage replay
  -> per-view Gaussian votes + majority vote + SP seed lifting
  -> shared-standardized CAC18+opacity FG/BG GMM + weighted kNN contrast
  -> integer s-t min-cut（保留 float 对照）与请求内 SP refinement
  -> 独立 Gaussian preview bitset/overlay
  -> Replace/Add/Subtract/Intersect Apply
  -> 一次 Apply 一条精确 selection swap Undo/Redo
```

## 真实源码接点

| 接点 | 已实现内容 |
|---|---|
| `src/SceneSnapshotAdapter.{h,cpp}` | 同步只读 GPU 快照；world position；宿主物理 scale、有效 quaternion 顺序、base color/higher SH；跳过 deleted/non-finite；保存 `localIndex -> snapshotRow` |
| `src/scene/SNSplats.h`、`src/SplatEditor.cpp` | `indexEpoch/contentRevision`；上传、排序、压缩、合并/过滤、主要几何/颜色/删除入口推进版本；node transform 另以矩阵 hash 验证 |
| `src/gaussian_selector/*` | CAC18、精确 3D kNN、igraph Leiden、SP graph、evidence、GMM、float Dinic min-cut、两种 refinement mode |
| `src/gaussians_rendering.cu` | standard stage 写 compact-stage identity sidecar；preview 在 staged color 上叠加，不修改属性或 flags |
| `src/render/gaussian_selector_coverage.cuh` | 单独捕获 opaque depth；按标准 tile 顺序重放；用排序后的 tile key 排除宿主零填充空 tile；复刻 Ring 与 disable-depthwrite early-stop 语义；完整视图累计 `A_all`，mask 只控制 `A_fg/A_bg` |
| `src/SplatEditor_draw.h` | standard renderer 同帧资源生命周期内 capture、coverage launch、event/pinned readback；ROI 外与其它节点仍参与 T |
| `src/GaussianSelectorRuntime.{h,cpp}` | cache、request/state、worker mailbox、stale reject、相机+整场景可见性 revision 的 per-view replacement、GPU/CPU budget、自动 3D ROI、preview/apply/history |
| `src/actions/GaussianSelectorAction.h`、`src/gui/menubar.h`、`src/gui/devFeatures.h` | `Dev&Debug` 旁的独立 `Dev` 按钮；GaussianSelector 折叠控件、ROI、FG/BG/erase、stroke undo/redo、Preview/Cancel/Clear、refinement、Apply |
| `src/update/inputHandlingDesktop.h` | Ctrl-Z/Y 优先处理 stroke history；没有 stroke step 时回到 editor model history |

## 身份与遮挡映射

持久身份是 `(nodeIdentity, indexEpoch, localIndex)`。快照携带 `SnapshotKey`，请求携带 `JobKey(snapshot, evidenceRevision, roiRevision, requestId)`。Evidence view key 还包含 viewport/view/proj、标准 raster 可见性设置，以及所有场景节点的 visibility/transform 与 Gaussian content/index revisions；该 revision 变化时丢弃旧 view votes，但目标属性未变时仍可复用 CAC/SP graph。

```text
node localIndex
  -> dLocalToSnapshot[localIndex]（deleted/non-finite 为 invalid）
  -> stage 时写 selectorStageRows[visibleSplatID]
  -> ordering/tile list 保存 stageIndex，可跨 tile 重复
  -> coverage 由 selectorStageRows[stageIndex] 回到 snapshotRow
```

非目标节点以及目标节点中 ROI 外/invalid 的项不写统计，但仍在相同有序 raster 中更新透射率。统计重放读取 Gaussian pass 前复制的 opaque depth，而不是已写过 GS depth 的最终 framebuffer。

## P1–P7 状态、最小剩余验证与风险

| 阶段 | 当前状态 | 本次可执行实现 | 仍需的最小验证 | 主要风险 |
|---|---|---|---|---|
| P1 | 完成 | feature 开关、固定 igraph 1.0.0、float cut、随机穷举 oracle | 依赖许可证归档 | 首次在线 FetchContent；igraph 构建较重 |
| P2 | 完成 | 真实 `SceneSnapshotAdapter`、transform/hash/epoch、CAC、两级图、预算预检 | 用实际带 node 非均匀变换的模型做 CAC 对拍 | 全量快照仍是同步 DtoH；超大模型可能有一帧停顿 |
| P3 | 主链完成 | pre-depth、sidecar、空 tile 校验、标准 alpha/early-stop 重放、launch/event/pinned readback、完整 coverage 分母 | 独立 CPU/GPU 数值 fixture：跨 tile、两 batch、外层 occluder、全遮挡 | 当前只支持 standard single-view 3DGS raster；尚无数值误差报告 |
| P4 | 完成 | 多视图 vote、SP seed、FG/BG GMM、calibration、min-cut、bitset | 实际场景选择质量矩阵 | 小 seed 集会进入明确的 degraded GMM |
| P5 | 完成 | 原生 action/toolbar、三种笔画、preview、四种 Apply、精确历史 | 人工 UI 验收 Replace/Undo/Redo 与下游复制/删除/导出 | 未做自动 GUI 测试 |
| P6 | 实现完成，压力验证不足 | worker、request 替代、stale reject、cache、可调 context 的自动扩张 ROI、溢出安全 CPU/GPU 总预算、实验收缩循环 | T34–T40 长期交互、ROI 扩张、cache equivalence、内存稳定性 | detached CPU worker不可抢占，只能丢弃结果；snapshot capture 同步 |
| P7 | 构建/启动验证完成，质量矩阵未完成 | ON/OFF Release、CTest、真实 NVRTC、修复并复测 Host/Device ABI | 用户真实数据集、active median/p95、GPU oracle、截图/质量检查 | 无真实模型输入时不能宣称选择质量或性能达标 |

## 固定图交互与 Algorithm 1 内部循环

- `InteractiveFullGraph` 是默认：每次新 evidence 都从有效 base scope 对完整固定 SP 图重算 GMM/unary/cut，因此新笔画可恢复上一轮漏选。
- `ForegroundRestrictedExperimental` 是规格 E 的显式工程变体：每个请求仍从 base scope 开始；第一次做完整图 cut，随后最多两次仅以当前 FG superpoints 重建 kNN 并求解，总计最多 3 次；FG/BG 模型始终使用本请求原始人工 evidence reservoir，不会因收缩删除 BG 样本；标签不变或活动集为空即停止。
- 该实验模式不是“已核实作者代码默认值”。作者完整工程循环仍未取得，因此 UI 和诊断明确标记 `Experimental/E`。
