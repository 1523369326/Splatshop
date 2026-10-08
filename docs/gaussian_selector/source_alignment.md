# GaussianSelector 来源与源码对齐

P=论文，B=作者实现笔记，S=Splatshop 当前源码，E=工程补全。已实际读取线上网页、其加载脚本和公开正文接口 `/api/blog/public/GaussianSelector-Implementation-Notes-What-Didn-t-Fit-in-the-Paper`。正文 `data.content` 与任务包 `blog.html` 的正文逐字符一致；作者为 Tiexin Li，接口显示更新时间 2026-08-05。defuddle 因 npm 代理拒绝连接无法运行，改用该网页自身的公开接口读取全文。

| 规则 | 来源 | 当前实现 | 偏差/验证状态 |
|---|---|---|---|
| CAC 为自身 3 主轴正负方向的 SH，共 18D | P Eq.(3)，B | `CanonicalAxisColor.cpp` | SH0、SH1、q/-q 单测通过；真实 node transform 模型对拍未运行 |
| base color 与 higher SH 不重复 DC | S | loader 已将 DC 转 base color；快照只把 higher SH 放进 `shRest` | 与 `getHarmonics`/stage 加法一致 |
| scale 已激活 | S | 直接读取设备 physical scale，不再 `exp` | 与 loader 的 `exp(scale_i)` 对齐 |
| quaternion 顺序 | S | device vec4 按 renderer 的 `glm::quat(q.x,q.y,q.z,q.w)` 转为有效 `{w,x,y,z}` | q/-q 测试通过 |
| node transform | S+规格 | graph position 使用 `transform_global`; SH 仍在 node-local radiance frame；transform hash 进入 stale 判定 | Gaussian 主轴沿 world 变换后再回 local 求 SH，node 线性变换在方向变换中抵消；真实非均匀 transform fixture 待补 |
| Gaussian 空间 kNN 与 weighted Leiden | P，B，E 参数 | 预算内精确 3D kd-tree；预算外 adaptive voxel mean nodes，再 exact kNN 与 igraph weighted CPM Leiden | 保留所有 Gaussian 成员；预算/确定性/broadcast 测试通过；40k/50k 为 E |
| SP mean pooling 后重建 SP kNN | P | `poolAndBuildSuperpoints` + `buildSpatialGraph` | 已实现 |
| alpha 已含 T | S | coverage 从 `exp(-4r)*(1-accumulatedAlpha)*opacity` 出发，终止状态跨 batch 保存 | 没有二次乘 T；标准 raster stop 与可选 evidence stop 分开；CPU/GPU 合成 fixture 通过 |
| coverage 分母覆盖完整视图 | P Eq.(6)，B，规格 | 所有有效 pixel 对 `A_all` 累计；同一证据可见性规则作用于分子和分母，FG/BG mask 只决定分子 | GPU 验证未绘制像素、FG/BG 对称；不把分母限定到 scribble/ROI pixel |
| evidence transmittance cutoff | B + E | 默认 .9、16 owners；UI 仍提供 0/.5/.9 | native 一个 Gaussian/pixel 一个 contributor；未复制第二个 Spark 实例的 48-layer peeling；可见质量分子分母一致 |
| support / center_visible | B + E | GPU 测量 FG/BG support、center-visible，使用 2% relative depth tolerance | 实际 GPU fixture 通过；support>=1、optional center gate 的组合规则为 E；depth-interval filter 未核实、未实现 |
| 空 tile 不产生虚假 coverage | S | coverage 用已排序 `tileIDs[first/last]` 验证零填充 `Tile` 是否真实属于当前 tile | 避免把真实 fragment 0 重放到所有空 tile |
| ROI 外/其它层仍遮挡 | 规格 | 它们保留 stage/tile/raster 合成，只以 invalid row 禁止 stats write | 已接真实执行路径；外层 occluder fixture 未运行 |
| pre-GS opaque depth | S+规格 | Gaussian pass 前 `kernel_gaussian_selector_capture_depth` 复制独立只读 depth | 不使用最终 framebuffer depth；NVRTC 启动验证通过 |
| compact/tile 身份 | S+规格 | `localToSnapshot -> selectorStageRows[visibleSplatID] -> tile stageIndex` | deleted/non-finite 为 invalid；epoch/content/transform stale reject |
| per-view vote 平票为 Unknown | P+规格 | 同 view mask 替换旧 evidence，多 view 严格多数 | tie 单测通过 |
| view evidence 随遮挡状态失效 | S+规格 | view key 包含相机/viewport、关键 raster 设置、场景节点 visibility/transform、Gaussian revisions 与其它几何 buffer identity | 目标未变时只清 evidence、保留视图无关 graph；未来新节点类型仍需接入 revision |
| 19D 双 GMM、median calibration | P Eq.(7–9)，B，E | diagonal GMM min(3,n-1)、EM<=80；blend confidence-weighted k=8 density，median calibration、stable softplus | singleton 靠 kNN；mix=.5/kernel 为 E；.95/.05 calibration 及小样本测试通过 |
| weak pseudo-BG | B + E | 无 hard BG 时最多 32 个 soft model samples，UI 标数量 | appearance-farthest 排名和 .1 权重为 E；作者 semantic SP 排名未核实；不是人工 hard BG |
| dynamic integer min-cut | B + E | 原生非递归 int64 push-relabel，按最大 capacity 和 overflow budget 动态缩放 | N<=9 与量化能量全枚举一致；不是 Boost 实现/量化参数的逐位复刻；float baseline 可切换 |
| result filters | B + E | FG-seeded graph components、oversized physical scale rejection | post-cut filters，不替代 CAC/Leiden/GMM/cut；具体规则和开关默认为 E |
| 有限 seed penalty=4 | E | 只给相反标签成本加有限 penalty | 不是 hard terminal |
| s-t terminal 方向与 Potts 边 | 图割规则 | source->node=BG cost，node->sink=FG cost；canonical edge 在 flow 中表达一次 Potts | 1/2 节点与随机 N<=10 穷举通过 |
| Preview 不写 `FLAGS_SELECTED` | 规格 | device preview bitset仅在 stage color 上混色 | Apply 前属性/flags 不变 |
| Replace 精确 Undo/Redo | S+规格 | Apply 保存 before selection mask；`kernel_swap_selection` 交换前后状态，只改 selected bit | 纯函数测试通过；宿主接线已构建，GUI 人工验收待做 |
| 缓存固定域交互重求解 | B + E | 每请求从 immutable base，冲突/不确定 SP 请求内细化，最多两次 cut | domain ordered localIndex content 与图参数进入 key；不复制整个编码；重复 solve 测试通过 |
| mixed FG/BG SP 局部拆分 | B + E | 最多 64 个冲突 SP，局部 Gaussian kNN + Leiden，重新 pooling/SP graph/GMM/cut | 20k 成员上限及 Leiden 后仍混合时按 FG/BG/Unknown 分组为显式 E；不修改 cached base；超预算拒绝错误 Preview |
| post-cut low-confidence SP split | B + E | abs(p_fg-p_bg)<.25，最多4个、每个<=20k；局部 Leiden、SP graph 重建、第二次 cut | 真实拆分/member budget 测试通过；局部更细 resolution 是 E |
| Algorithm 1 foreground-restricted | P，细节未全公开；E 补全 | 可选再至多2次 FG graph cut，保留人工 BG reservoir | 独立 Experimental 模式；不是作者两阶段 SP split；需要真实 BG strokes |

## 对齐边界与未核实内容

已采用笔记公开的数值和工程方向，并对其未公开细节明确标 E。未实现 Spark 的第二实例/48 层 depth peeling；native replay 每 Gaussian/pixel 单 contributor，默认16-owner预算。depth-interval filter 的完整规则仍未核实且未接入。support gate 的组合判定、unary_mix 数值、局部 Leiden 约束、semantic pseudo-BG 排名、component/oversized 默认阈值和整数精度仍为工程补全，不能声称与作者未公开代码一致。

一手入口：论文 [arXiv 2608.01492](https://arxiv.org/abs/2608.01492)，[项目页](https://white-yc.github.io/GaussianSelector/)，[作者实现笔记](https://lllab.org/blog/GaussianSelector-Implementation-Notes-What-Didn-t-Fit-in-the-Paper)。
