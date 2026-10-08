# GaussianSelector 已知限制

- 新 ROI UI 使用屏幕矩形内 Gaussian 中心投影来限定求解域；绘制的 mask 被裁剪到框内，coverage 分母仍覆盖完整视图。相机/ROI scope 改变后重建范围并清空旧 votes；未声明此屏幕 ROI 规则为论文原始默认。真实场景的框选、擦除与 Apply 全流程仍需人工验收。

- coverage 只接到 standard、single-view、staged/tiled 3DGS raster kernel；SoA、bandwidth、fragment-intersection、sort-free、RaDe、Heatmap/Core-Tail、Solid、dual culling 与 multi-view 模式在 UI 中明确拒绝。
- evidence 默认 T=.9、16 owners；只减少后层种子污染，不直接禁止 Unknown 后层被 GMM/graph cut 判为 FG。更高 stop 可能漏掉薄结构。native contributor replay 不等同于 Spark 48-layer peeling 的逐位复刻。
- cut 前冲突拆分、cut 后最多4个低置信度 SP split、混合密度、adaptive voxel、整数 cut、component/oversized filters 已实现。混合比例、局部 resolution、pseudo-BG 排名、filter 判据等为明确 E；详见 parameters.md。support/center 已测量（2% depth tolerance），但作者完整组合规则及 depth-interval filter 仍未核实；depth-interval 未接入。
- GPU snapshot 的 DtoH 与 row 组装当前同步发生；CAC/kNN/Leiden/GMM/cut 在 worker 中异步。超大但仍在 1 GiB 快照预算内的模型可能产生一次 UI 停顿。
- 自动 ROI 由高置信 FG 三维包围盒加默认 `max(20% FG diagonal, 2% scene diagonal)` context 构成，并保留全部人工 evidence 节点；toolbar 可调整/重置两项 padding 后重新 Solve。这是规格 E 策略，不是作者默认值；目前显示 scope 数量，但没有可拖拽的 3D ROI gizmo。
- 500k 现在是旧 runtime 的 full-scope warning，不再是当前屏幕 ROI 的固定候选上限。snapshot 仍受 1 GiB 预算限制；graph 默认最多50k mean nodes，体素 pooling 保留所有 Gaussian，不随机抽样。超预算明确拒绝。
- CPU worker 不能中途抢占；Cancel/新 request 通过 request id 丢弃迟到 publication。长期连续替代任务的 CPU 峰值尚未压力验证。
- `contentRevision` 已覆盖当前主要上传、重排/压缩、合并/过滤、几何、颜色和删除路径，node transform 由矩阵 hash 独立检查；仍建议新增集中式属性 mutation API，避免未来新增编辑 kernel 忘记推进版本。
- evidence visibility revision 已覆盖现有 Gaussian revisions、全部节点 visibility/transform、triangle/point buffer identity 与关键 standard-raster 设置；未来新增会写 pre-GS depth 的节点类型或原地修改 triangle/point buffer 时必须显式扩展 revision 通知。
- 默认 cut 为非递归 native int64 push-relabel；动态缩放/量化能量的最优性已用穷举验证。原始 float 能量与量化能量不是同一目标，精度1e6为E；float Dinic 保留作对照。大 SP 图最坏耗时仍需实测。
- `ForegroundRestrictedExperimental` 是明确的 E 补全，不等同于已取得作者源码的 Algorithm 1 完整实现。
- 已有实际 coverage kernel 的 GPU 合成跨 batch 数值 fixture；fixture 简化了 StageData packing/decoder，不替代真实 renderer 的 ABI、跨 tile 或真实场景对拍。真实模型质量、active 性能与长期内存结论仍需验证。
