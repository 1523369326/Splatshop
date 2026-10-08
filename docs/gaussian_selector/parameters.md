# GaussianSelector 参数

当前默认值都在 `src/gaussian_selector/SelectorConfig.h`。P=论文、B=已读取的作者实现笔记、E=规格/显式工程补全。工程值不声明为作者默认值。

| 参数 | 默认 | 来源/说明 |
|---|---:|---|
| Gaussian/SP kNN | 8 | P §4.1 |
| distance gate quantile | 0.95 | E |
| spatial/color/opacity weights | 0.35/0.4/0.25 | E (user-tuned) |
| Leiden objective | CPM | E，作者默认未核实 |
| Leiden resolution/beta/iterations/seed | .05/.01/2/42 | E |
| GMM components | min(3,samples-1) | B；singleton 使用 1 个安全占位分量，混合模型改用 kNN 对比，避免零分量 GMM |
| EM iterations/tolerance/variance floor | 80/1e-4/1e-3 | 上限 80 为 B；tolerance/floor 为 E |
| standard deviation floor | 1e-6 | E |
| seed target confidence | .95 | P/E calibration；不同于 coverage threshold |
| finite seed penalty | 4 | E |
| pairwise lambda | 1 | E |
| coverage Q10/floor/min visible mass | .10/.10/.01 | B 描述 adaptive 10%-quantile；对 FG/BG 分别取非零 ratio Q10 并保留 .10 floor 是 E |
| evidence transmittance stop / owners | .9 / 16 | B；native replay 每 Gaussian/pixel 一个 contributor，不模拟 Spark 的重复 depth-peeling layers。Standard 按钮同时清 T 与 owner cap |
| center relative depth tolerance | .02 | B；GPU 测量中心在 mask 且相对最前 contributor 深度误差不超过 2% |
| support gate / require center | 1 pixel / disabled | E；B 未公开两种测量的完整组合规则。optional center gate 位于 advanced 参数 |
| split conflicting SP | enabled | B 的 cut 前局部拆分方向；每次请求从 cached base 开始，不改变场景编码 |
| max conflict SP/members | 64/20k | B 描述 cut 前最多 64 个 SP；本项目 cut 前成员上限 20k 是 E（B 的 20k 描述属于 cut 后低置信度 split）。超限明确拒绝 Preview |
| post-cut SP/members/gap | 4/20k/.25 | B；最多一轮，按 abs(p_fg-p_bg) 排序后重新 pooling、SP graph、contrast、cut。局部 resolution=max(.2,4*base) 是 E |
| weighted appearance kNN / GMM mix | 8 / .5 | k=8 为 B；标准化 19D kNN ball density、radius floor 与 .5 mix 是 E，原始 unary_mix 未核实 |
| weak pseudo-BG max/confidence | 32/.1 | 32 为 B；appearance-farthest、无 FG 支持的 SP 排名与 .1 confidence 是 E；不制造人工 BG hard seed，UI 声明实际数量 |
| adaptive graph target/max/probes | 40k/50k/20 | probes=20、增长1.5、收缩.8、目标带[.75*target,max] 为 B；40k/50k 为 E，可调。预算内直接 Gaussian exact kNN，预算外保留全部成员，经 voxel mean graph + Leiden 映回 |
| integer cut precision | 1,000,000 | B 描述动态缩放的整数 push-relabel；native int64 solver 与精度为 E，exact quantized optimum，不声称与作者 Boost capacities 逐位相同 |
| result component / oversized filter | enabled/enabled | B 描述两种控制；保留含 FG hard seed 的图连通分量、max physical scale > 8*ROI median 为 E，可关闭对照 |
| CPU/GPU incremental budget | 1 GiB/256 MiB | E；snapshot + 固定 graph workspace 与 GPU identity、pre-depth、masks、7 张 mass/support/center、preview 做预算预检 |
| full-scope ROI warning | 500k Gaussians | E；无显式框的旧 runtime 大域先 lifting，再自动 3D ROI；当前屏幕 ROI 候选上限按 CPU snapshot 预算计算，图节点由 voxel budget 限定 |
| ROI FG/scene padding | .20/.02 | E；toolbar 可调并可重置；下次 Solve 重算 scope，ROI 外仍参与遮挡 |
