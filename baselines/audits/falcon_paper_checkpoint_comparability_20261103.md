# Falcon 论文指标、发布检查点与子集评测边界

核查日期：2026-11-03。只读核对原 PDF、固定源码、官方 Drive 模型说明及检查点元数据；未运行模型、未改官方源码、未触碰 job28088。

## 原论文表 II：Social-HM3D

已将用户提供的原始 PDF 第 6 页重新渲染并目视核对；纸面页码 9127。原 PDF SHA256 为 `77635dd2eb491cf49890c19e2115197e527532ae2cfe0ea40768b6778af0c088`。与 `research/falcon_source.txt` 的数字一致。以下全部为百分比，格式为均值±标准差：

| 方法 | SR（原文 Suc.）↑ | SPL ↑ | STL ↑ | PSC ↑ | H-Coll ↓ |
|---|---:|---:|---:|---:|---:|
| A* | 46.14±0.7 | 46.14±0.7 | 46.12±0.7 | 90.56±0.2 | 53.50±0.9 |
| ORCA | 38.91±0.1 | 38.91±0.1 | 38.44±0.1 | 90.55±0.4 | 47.52±1.7 |
| Proximity-Aware | 20.11±1.3 | 18.57±1.9 | 19.51±1.5 | 92.91±0.5 | 33.99±0.7 |
| Falcon | **55.15±0.6** | **55.15±0.7** | **54.94±0.7** | **89.56±1.4** | **42.96±1.1** |

原论文**没有 Total 列**。Falcon 在该表的成功率/效率最佳，PSC 和 H-Coll 并非各方法中的最佳值，不能表述为全部指标最优。

[原 PDF 页面的核对图](falcon_original_pdf_page6_20261103.png)；[结构化数字与证据](falcon_paper_checkpoint_comparability_20261103.json)。

当前官方 `falcon_evaluator.py:375–388` 输出 SR、SPL、PSC、H-Coll 和 Total，数值尺度为 0–1。其中 `Total = 0.4 × SR + 0.3 × SPL + 0.3 × PSC`，不含 H-Coll。将论文 Falcon 均值代入可计算 0.65473（65.473%），但这只是按代码公式派生的数，**不是论文报告的 Total**；不能从边际标准差推导其标准差。当前发布默认配置不启用 STL，故当前四项加 Total 不能声称覆盖论文全部五项指标。

## noaux_25 属于哪个模型

官方 README 的 Falcon 评测命令使用 `social_nav_v2/falcon_hm3d.yaml`，该 YAML 明确指向 `pretrained_model/falcon_noaux_25.pth`。实时读取的[官方 summary.txt](https://drive.google.com/file/d/1U3IdPQeG6vV7XzbU8iH13A-DpLloRY3f/view)将它标为 “A full-version Falcon model without auxiliary tasks”，并指定直接推理用途；另一文件 falcon_pretrained_25.pth 用于继续训练。

可确认的身份是：**作者发布并由官方评测配置选择的完整 Falcon 推理权重**。不能仅凭 noaux 文件名把它归入表 III 的 PointNav“未用辅助任务训练”行。表 III 该基线的 SR/SPL/STL/PSC/H-Coll 为 40.94/34.14/11.50/90.82/53.54；完整 SPM+SCP 行为 55.15/55.15/54.94/89.56/42.96。

本地文件 SHA256 为 `ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca`。通过只读、惰性占位类解析 data.pkl（没有导入模型、没有重建张量），观察到：

- agent_0 的 state_dict 共 179 个键，没有 aux/human/future/people 相关键。
- 内嵌 config 仍保留三个辅助损失：计数、当前位置、未来轨迹；各 loss_scale=0.1。
- 内嵌 config 包含未来轨迹社会惩罚 `trajectory_cover_penalty=-0.0005`。
- 这些与发布说明共同支持“辅助头不参与推理，策略来自完整 Falcon”的解释；它们不是完整训练历史的证明。

无法绑定到论文中的具体训练 seed、确切训练步数或三次运行中的某次，也不能确定文件名 25 的 epoch/seed 含义。虽然内嵌 seed=100，其配置还混有 `multi_agent_debug` 数据路径、ORCAEvaluator、evaluate=false、total_num_steps=1e8，而 extra_state 只有 step=256、wall_time≈29.83s；该元数据明显不足以充当论文 10M-step 训练日志。当前源码训练 YAML 还写 15M steps，并注释 exact steps 12560368，和论文 10M steps 不一致，必须保留版本/训练历史不确定性。

[元数据](falcon_noaux_checkpoint_metadata_20261103.json)由 [只读解析器](../scripts/audit_falcon_checkpoint_metadata.py)生成。它只将 pickle 记录解码为元数据，不实例化上游类，不能验证训练发生过。

## 当前实际评测依据

固定源码 commit：`d15ef512c4b1451f9daeaf6d4d7ecc702ed57430`；原仓库无修改。官方评测 `use_ckpt_config=false`，所以实际依据是发布 YAML 及运行时的 resolved_config，不能使用检查点的旧配置替代。

嵌入配置与成功 smoke 保存的官方 resolved_config 有 54 个字段差异，主要为训练辅助损失/传感器、reward/STL、旧路径与 evaluator。两者一致的关键字段包括 seed100、500 步上限、0.2m 成功阈值、HumanCollision、PSC 的测地距离开关、四动作参数、ResNet50/两层 LSTM/512 维结构。完整差异在 [配置对比](falcon_checkpoint_eval_config_diff_20261103.json)。

174 条运行相对官方评测配置另外记录了：9 个 content_scenes、val 绝对路径、4 个并行环境（官方 8）、可审计的日志 evaluator 及输出/权重路径。eval split 从 train 切换到 val 是官方评测器本来的行为；4 环境和场景筛选则是必须报告的执行差异。随机动作采样与场景分配意味着改变并行数也可能改变随机序列。

## 最终报告应怎样表述

当前任务是**官方发布 Falcon 单检查点在 Social-HM3D 公开 val 的 9 场景/174 episode 子集评测**。公开 val 已盘点为 56 场景/1087 episode；174 条约占 16.01%，按现有官方 minival 几何覆盖选取，并非随机抽样。以上数量来自发布数据的实盘清单，不能声称是论文正文明确列出的测试集数量。

如果完成并通过独立 coverage 验证，可以报告该子集的实际 SR/SPL/PSC/H-Coll/Total；不能拿它与论文完整评测表计算“复现差距/超过论文/SOTA”。即使后续完成 1087 条，仍是公开单检查点评测，不是三次不同随机种子实验的均值±标准差重建。当前审计没有读取或认定 job28088 的最终结果。

