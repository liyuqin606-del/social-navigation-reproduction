# 社会导航复现与实验归档

2026-11-08 私有快照。**实验已按用户要求暂停，没有自动续跑。** 本仓库保存复现脚本、上游版本、已接受的输出、失败记录，以及明确分开的探索性实验；不是四篇论文均已复现完成，也不是 SOTA 声明。

论文阅读笔记：[social-navigation-reading-notes](https://github.com/liyuqin606-del/social-navigation-reading-notes)。

## 当前复现状态

| 工作 | 已完成内容 | 尚未完成／限制 |
|---|---|---|
| Falcon | 发布检查点评测；174 回合非随机子集独立验证；全量任务中接受 471/1087 回合、24/56 场景 | 剩余 616 回合暂停；没有训练或多种子论文复现 |
| NavThinker | 官方实现诊断、导入兼容性排查、随机权重合成前向检查 | 截至 11 月 3 日核查，检查点缺失且策略入口存在 RSSM 接口问题；无 benchmark 成绩 |
| DPed-VLN / DPet | 数据与公开实现材料核查 | 当前快照未取得可完成作者基线复现的完整代码；无 benchmark 成绩 |
| Social-WM | 论文与公开材料核查、参考实现差异分析 | 核查时未找到官方代码、权重及训练轨迹；未运行 Social-WM |

174 回合子集：SR **55.75%**，SPL **50.89%**，PSC **89.44%**，H-Coll **40.80%**。它使用 seed100、4 个环境，来自最初可用的 9 个场景；不是完整基准，不能与论文表格作严格等价比较。

174 子集与 471 已接受回合是不同评测组织，存在重叠，**不可相加**。全量分片改变随机流；取消后的部分记录与验证失败的作业不计入 471。

## 文件入口

- [原始历史复现状态](baselines/REPRODUCTION_STATUS.json)：保留 11 月 4 日的原记录，内含当时状态，不是实时调度信息。
- [暂停记录](research/PAUSED_20261108.json)：当前执行状态。
- [174 子集结果](baselines/results/falcon_subset174_20261103.md)、[逐回合证据](evidence/subset174/)。
- [471 回合分片证据](evidence/accepted471/)、[无重复覆盖核对](evidence/accepted471/COVERAGE.json)。
- [复现和验证脚本](baselines/scripts/)、[诊断审计](baselines/audits/)、[环境记录](baselines/prep/environment_evidence.json)。
- [上游仓库与固定提交](UPSTREAM.json)。官方源码不在此完整重分发。
- [权重编辑工作台](weightlab/README.md)、[事件导航原型](eventnav/README.md)：工具／研究原型，独立于论文复现成绩。

## 探索性实验单独归档

`research/` 下的理论、CPU 小实验和 Falcon 控制器实验均为开发研究，不计作正式复现完成数量。

| 实验 | 结果及边界 |
|---|---|
| Falcon 接口迁移探针 | 已验证动作修改与记录接口；不是性能提升 |
| SCET 有限模型 pilot | 保留小模型结果与负结论；不是机器人 benchmark |
| SCET-R 首轮短动作诊断，28437 | 4 个开发案例、20 条完整轨迹、2787 步；3 个碰撞基线救回 0 个 |
| 观测驱动控制器 A，28439 | 4 个开发案例、20 条完整轨迹、3113 步；三个候选均 0 救回、0 成功退化；均仍为 1/4 成功 |
| 控制器 B / 运动残差原型 | 本地代码／检查阶段，未提交 GPU，已暂停 |

28439 的本地 owner 因作业离队后的状态查询异常留下失败记录；远端实际 `COMPLETED / 0:0`。此次暂停归档时完成 7 项输出 SHA 回收及本地科学验证，保留[原 owner 失败记录](evidence/experiments/28439_failed_owner_preserved.json)和[独立恢复记录](evidence/experiments/28439_recovery.json)，没有改写历史或重跑。

## 使用与依赖

先阅读 [UPSTREAM.json](UPSTREAM.json) 与环境记录，再按各脚本 `--help` 准备上游源码、合法获取的数据和检查点。代码是历史运行快照，部分入口保留机器绝对路径；需按自己的环境调整并重新生成冻结计划。**本仓库不是解压即可直接提交的算力配置。**

本地快照验证无需模型或 GPU：

```bash
python3 tools/verify_snapshot.py
```

不包含 HM3D/Matterport3D 场景、原始任务数据包、机器人资产、模型权重、Python 运行时、SSH 配置、访问申请、签名文件或私钥。完整冻结运行目录和视频保留在本地；仓库中的摘取证据不能替代全部原始运行包。数据与资产必须按各自许可获取。

`EXPORT_PROVENANCE.json` 记录逐文件来源及 SHA-256。导出文件保持原内容；其中的历史日期、路径和未完成标记没有被改写成当前成功状态。`SHA256SUMS` 校验整个发布快照。许可与来源见 [NOTICE.md](NOTICE.md)。
