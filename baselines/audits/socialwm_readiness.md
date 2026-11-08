# Social-WM 复现准备核查

核查日期：2026-11-03。目标：[Social-WM: Safety-Aware Latent World Models for Robot Social Navigation，arXiv v2](https://arxiv.org/html/2609.40177v2)。**当前到达材料固定与代码参考核验阶段；没有运行 Social-WM、加载其检查点或取得论文跑分。**

## 官方材料现状

在本轮检索范围内，未找到可归属于作者的 Social-WM 源码、检查点、采集轨迹或可执行训练/评测入口；这是截至核查日的有界检索结论，不是证明它们不存在。

已检查：论文全文与 arXiv 摘要页的外部链接，GitHub 仓库搜索 `Social-WM` 与 `socialwm`，精确标题与作者/邮箱组合搜索，以及作者所属 [WiNS Lab 团队页](https://winslab.cse.lehigh.edu/team.php)、[项目页](https://winslab.cse.lehigh.edu/project.php)、[新闻页](https://winslab.cse.lehigh.edu/news.php) 和[导师论文页](https://www.cse.lehigh.edu/~chuah/publications.html)。论文 HTML 中的 GitHub 链接均属于 arXiv/LaTeXML 工具。团队页指向第一作者主页 `https://www.cse.lehigh.edu/~zhzc21/`，该地址在网页工具和直接请求中都返回 403，因此这条发布渠道没有读到；未尝试绕过。未将第三方 alphaXiv 的通用“复现”按钮、论文列表或同名项目当官方实现。

## 已落地的可复查材料

- `baselines/prep/socialwm/sources.json`：六份官方网页快照的字节数、SHA256、来源，以及第一作者主页的 403 记录。论文全文另有本地纯文本提取。
- `baselines/prep/socialwm/reproduction_contract.json`：论文确定的架构、训练数据与物理输入协议；20 项未公开或未能唯一确定的参数保留 `null`。它是准备契约，不是作者配置。
- `baselines/scripts/check_socialwm_readiness.py`：纯标准库、离线输入检查，分 `--mode eval` 与 `--mode train`，不导入 torch，不下载、不提交作业。缺输入返回退出码 2；通过也只代表材料检查通过。
- `baselines/prep/socialwm/readiness_check.json`：本轮真实执行结果，评测材料未就绪；14 个已固定源文件 SHA 全部通过。
- `baselines/prep/socialwm/upstream_lewm_reference/`：通过 [LeWM 论文自身的代码链接](https://arxiv.org/html/2603.19312v3) 确认的上游 `lucas-maes/le-wm` 参考快照，commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，8 个文件、34,554 字节，保留 MIT LICENSE。每个文件与 GitHub 原始 blob SHA 完全匹配，4 个 Python 文件通过 AST 语法检查。**这是 LeWM 参考材料，不是 Social-WM 仓库或可运行 Social-WM。**

复查命令（在 VLN 根目录）：

```sh
.venv/bin/python baselines/scripts/check_socialwm_readiness.py --mode eval --output baselines/prep/socialwm/readiness_check.json
```

## 从论文中确定的实验要求

|项目|明确报告的值|
|---|---|
|输入/动作|前向 RGB、机器人自身 pose/odometry、目标；连续机体坐标前向/侧向/yaw 三分量|
|视觉表征|冻结 DINOv2-small，224×224，去掉 CLS，空间 patch 池化为 16 tokens × 384维|
|预测/规划|6 层 Transformer；历史 T=8、未来 H=8、候选 K=8；默认 CVAE；每步只执行选中动作块的第一步|
|训练|AdamW；ID head 随机初始化在预训练 trunk 上，head LR 是 trunk 的10倍|
|采集|Falcon 的 Social-HM3D train 中 200 scenes、11,676 episodes、463,645 transitions；18.1% 为 blocked transitions|
|名义/实际动作|WM 以名义动作预测实际观察到的未来；ID 以连续位姿恢复的实际实现动作作监督|
|采集 keep-out|采集时拒绝会进入行人的命令，记录停止或受限的实际后继状态；不能把这种受限转移换成理想运动|
|主评测|Social-HM3D final position goal；Social-MP3D 零样本迁移；SR/SPL/PSC/H-Coll|

这些细节来自论文 III-A–F、IV-A–B。Falcon 的 episode JSON 只描述任务，不包含上述 RGB/位姿/名义动作/实际动作序列；即便共享场景资产准备完成，也仍不能直接训练 Social-WM。论文的 200 个训练场景子集、轨迹行为策略与 split IDs 均需进一步确认。

## 阻碍忠实运行的具体缺项

推理侧仍缺：官方结构与 checkpoint、动作量纲/归一化/范围、控制周期与 odometry 到动作的变换、残差距离的量纲和通道、阈值 τ、没有可行动作时的 fallback、仿真碰撞/门控配置、确切评测 episode/seed/count、停止规则/成功距离/步数上限、依赖版本。公式(9)写一般范数，IV-G 诊断又写 xy 范数，不能自行认定 yaw 是否参与规划安全分数。

从头训练额外缺：原始采集轨迹或精确采集脚本，keep-out 几何/半径，采集策略/seed、场景和 split 清单，绝对 LR、batch、epochs/steps、scheduler/weight decay，以及四个 WM loss 权重和 CVAE β。论文给出的层数和维度也不足以唯一恢复 attention heads、MLP、dropout、ID 输出分布或 CVAE latent size。上述值没有从其他论文或 LeWM 默认值借填。

代码级检查发现，上游 LeWM 的 `jepa.py` 直接使用 CLS token、默认 history=3、终点 embedding MSE 代价；其默认训练 YAML 为192维。这与 Social-WM 的16个空间 tokens、T=8、384维、realizable ID/CVAE 与安全筛选不同。故不能直接运行 `python train.py data=pusht`，或加载 LeWM 的 PushT/TwoRoom 权重，便称为 Social-WM 复现。已有上游快照只能供之后的结构对照或明确标注的独立实现参考。

## 后续结果必须守住的口径

论文表 I 主结果是 final-position-goal：HM3D 的 SR/SPL/PSC/H-Coll 为63.77/49.62/94.37/21.67，MP3D 为54.15/44.32/93.95/23.34（均为百分比）。表 II 的 position-subgoal 68.65/58.72使用了参考路线中间信息，应单列；不能用它当同输入条件下的主 baseline。

本工作只有采集 keep-out 被明确描述；部署侧是模型预测残差筛选。真实人体状态只可进入环境/审计，不可静默加入模型输入或测试真值安全闸。连续三分量动作也不能无说明改为 Falcon 的四离散动作再直接比较论文表格。论文 A5000 的51ms/step不是本项目3090的实测结果。

下一可执行关口：官方 code/config/checkpoint 或足够完整的释放材料出现后，先填写契约并核验资产，再由统一环境执行短回合和正式评测。目前无源码/模型可执行，因此没有为本项申请 GPU、SSH、安装共享环境、下载大型权重，或联系作者。
