# 作者实现笔记对齐与当前优化

已通过 [作者页面](https://lllab.org/blog/GaussianSelector-Implementation-Notes-What-Didn-t-Fit-in-the-Paper) 自身公开接口读取正文全文，并确认与任务包保存正文逐字符一致。页面是客户端渲染；普通网页抓取只返回空壳。defuddle 下载受本地 npm 代理阻断，最终使用公开接口读取。

## 本轮已实现

1. **永久保存 pixel 终止状态**：coverage 遇到 pre-GS opaque depth 或透射率终止后，不再在后续 staging batch 复活。ROI 外/其它节点的 invalid-row contributor 仍能遮挡。CPU oracle 与 CUDA 共用状态规则。
2. **笔记 evidence visibility**：默认 T stop=.9、16 owners；可切换标准 replay（同时取消 T/owners）、.5、.9。所有像素采用相同规则统计总质量，FG/BG 分子取同一质量的 mask 子集。新增 support、center-visible GPU 测量，中心深度容差2%。不是第二个 Spark 实例/48 层 peeling 的完整复制。
3. **cut 前冲突 SP 拆分**：FG/BG 混在一个 SP 时，局部 Gaussian kNN + weighted Leiden 后重建 pooling、SP kNN、GMM 与 min-cut。Leiden 仍混合时，明确的工程约束把该 community 内 FG/BG/Unknown 分开，Unknown 不成为 seed。最多 64 个 SP、每个 20k 成员；超限拒绝 Preview 并提示缩小 ROI/提高 Leiden resolution。
4. **复用不可变编码**：无冲突的重复 solve 通过 shared immutable encoding 避免复制整个 CAC18/partition；有冲突时仅建立请求级 refinement。输出仍是原 Gaussian 顺序的 mask，Apply/Undo 继续使用 node/localIndex/epoch。
5. 保留用户权重 .35/.40/.25；UI 显示实际权重、FG/BG seed 数、冲突拆分数及求解 SP/edge 数。
6. **预算控制**：20次以内 adaptive voxel resolution search，按1.5增长/.8收缩、目标带[.75*target,max]；预算内 exact Gaussian kNN，预算外 voxel mean kNN + Leiden，不随机采样或丢弃 Gaussian。40k/50k 节点预算为E。
7. **混合对比模型**：EM<=80、GMM min(3,n-1)，singleton 改用非参数估计；confidence-weighted k=8近邻密度与GMM混合，再median calibration。mix=.5、kNN球密度/floor为E。无hardBG时最多32个weak soft samples；appearance-farthest排名与confidence=.1是E，不冒充作者未公开的semantic ranking。
8. **第二次cut**：对 abs(p_fg-p_bg)<.25 的SP，最多4个、每个<=20k，局部Leiden细化后重新pooling/SPgraph/contrast/cut。更细resolution=max(.2,4*base)为E。cached base保持不变。
9. **整数求解和结果过滤**：native int64 push-relabel 动态缩放，精确优化量化能量；整数精度1e6为E。保留含FG hard seed的连通分量、拒绝physical max scale >8*ROI median的Gaussian，这两个具体判据及默认enabled为E，可在主面板关闭对照；没有替代论文主体链路。

2026-10-01：主程序Release和核心测试构建成功；量化能量穷举、体素预算/成员映回/确定性、单样本混合对比、四个真实post-cut拆分、pseudoBG声明、component/oversized开关、选区Undo回归通过。CUDA实际coverage fixture新增owner cap、support及2%center-visible检查，完整renderer NVRTC编译通过。尚未操作用户真实模型。

## 尚未实现或未完整核实

上述工程方向已接入，但作者的原始代码和部分默认值没有公开。未实现Spark第二实例/48-layer depth peeling；native replay一个Gaussian/pixel一个contributor，16-owner上限也不会超过48层。depth-interval filter完整规则未核实，尚未接入。support与center的组合判定为E（support>=1、center gate默认关闭）；混合kernel、局部Leiden约束、semantic pseudoBG、filters具体判据与整数精度仍不保证和作者逐位一致。ForegroundRestrictedExperimental仍是另一个独立工程循环，测试笔记配置应关闭它。

## 质量边界

二维ROI仍包含后方Gaussian；T控制种子可见性，Unknown标签仍由混合contrast与cut推断。T=.9可能漏选薄结构，component过滤可能移除未画FG的分离花瓣，scale过滤可能移除合法大Gaussian。先点Use author-notes profile，画真实FG/BG、Preview、Apply、Undo；漏选时分别关闭filters对照，再比较0/.5/.9，不应把所有误差归给T。

验证结果见 [validation.md](validation.md)。
