# Falcon managed eval spec 准备

已提供 `baselines/scripts/make_falcon_eval_spec.py`，仅生成spec，不调用SSH、不freeze、不启动任务。当前三份可审阅草稿：

|草稿|回合/场景|默认并行环境|GPU/时限|输出预算|
|---|---:|---:|---|---:|
|`falcon_eval_subset174_spec_layout_v2.json`|174/9|4|1×3090，1小时|128 MiB|
|`falcon_eval_full1087_spec_layout_v2.json`|1087/56|8|1×3090，1小时|576 MiB|
|`falcon_eval_shard00_spec_layout_v2.json`|130/7|1|1×3090，1小时|128 MiB|

草稿位于 `baselines/prep/`。输出预算是按每回合最多500步、每条动作记录1KiB加32MiB元信息估算，再向上取整到64MiB；它不是测量结果。正式运行后根据174子集的时间和实际输出大小更新full计划。每job科学超时3420秒、独立验证120秒，并为收尾留出时间；没有保证1087回合一定能在1小时完成。超时留下partial日志，不能当完整benchmark。

## 入口

```sh
.venv/bin/python baselines/scripts/make_falcon_eval_spec.py \
  --manifest baselines/prep/falcon_val_hm3dminival9_manifest.json \
  --deadline SERVER_LOCAL_FUTURE_YYYY-MM-DDTHH:MM:SS \
  --output baselines/prep/FRESH_SPEC.json
```

full使用 `falcon_full_val_20261103_manifest.json`；单个shard可直接传其manifest，或传 `falcon_full_val_shards_20261103.json --shard-id 00`。支持 `--num-envs` 覆写为1/2/4/8，环境数不能大于场景数。`--readiness-file` 可重复传入本地env-readiness/pip-freeze文件，它们会进入freeze集合；`--extra-remote-inputs` 可绑定对应远端文件的准确path/bytes/SHA。`--env-json` 可传非敏感库路径等环境变量；不复制密钥。`--scenes-dir` 和 `--data-path` 可明确覆写远端运行路径。

v2路径已与 `remote_full_val_prepare.py` 和实际转移包目录对齐：174保留repo内原minival场景根；full/shard使用 `ROOT/fullval_assets/scene_datasets`，在入队前绑定 `ROOT/inputs/falcon_full_val_geometry_selected56_20261103.tar.gz` 的1,959,090,598字节和SHA `1258c805...`。这是保持文件内容不变的56场景派生选择包，原3.5GB官方archive仅作来源追溯，不谎称已上传。full episode tar绑定 `ROOT/inputs/full_val_manifests/falcon_full_val_20261103.tar.gz`。shard也从已解出的full episode根加载，按精确content_scenes过滤并核验各原始shard SHA，因此不要求未上传的八份重复shard tar。科学来源manifest均未修改。旧 `*_spec_draft.json` 已标记SUPERSEDED_DO_NOT_FREEZE；使用v2或重新生成。

generator必须获得显式server-local deadline。当前三个草稿使用2026-11-03T18:00:00供审阅，freeze前需按实际服务器时间刷新，不能反复重用旧deadline或旧run_id。run_id含UTC时间和随机后缀，总长满足Slurm63字符约束。

## required_outputs 实际核查

managed worker已先创建job输出目录，科学入口使用其新建的 `eval/` 子目录，满足runner拒绝覆盖非空目录的要求。15个固定必需文件包括：run_status、run_manifest、原始episode manifest、映射后的精确expected episodes、解析后的YAML/JSON、派生evaluator/patch/manifest、逐回合JSONL、actions、官方rounded result、unrounded aggregate、completion marker、validation。它们是成功运行后确实生成且用于独立验收的文件。

**没有把 `habitat.log` 加为必需文件。** 官方 `logger.add_filehandler` 只在训练 `_init_train` 内，eval即便设置log_file也不保证创建它。错误地要求该文件，会把成功评测判为缺输出。评测stdout/stderr已由managed `scientific.log`回收。

动态TensorBoard事件名不列required_outputs；关键数值已有逐回合和聚合JSON。异常时 `failure_traceback.txt` 是可选文件，不强行要求成功运行也存在；异常同时打印进scientific.log，managed日志仍回收。中断时各个已存在的固定输出会纳入失败记录，不能据partial均值报告完整指标。

## 实际提交前的注意项

- env仍需能在计算节点加载libEGL并通过Falcon checkpoint/EGL smoke。login节点缺libEGL不能据此断言GPU节点已可用，也不能把静态spec当运行通过。
- owner在sbatch前验证两份checkpoint、Falcon源码archive、机器人archive、实际转移的episode archive与对应mesh archive的远端SHA。它不解压数据。174需要保留minival的val目录aliases；full/shard使用独立fullval_assets根，不能误指向仅有9场景的原repo数据目录。
- runner在模型运行前继续核查实际episode shard SHA和精确keys、mesh/navmesh文件，并把模型/场景SHA记录到结果；mesh包元信息由此前审计取得，没有为了生成spec再次读取数GB包。
- subset174使用4env会改变相较原8env的随机流；full默认8env更接近官方设置。逐shard重启也改变随机流。所有这些配置差异都应报告。
- 完整val的1087个回合属于单个已发布checkpoint的评测，仍不等于论文的三个独立训练seed。合并shard时必须验证复合keys的无重复完整并集、按episode聚合，不能等权平均各shard均值，也不能把已运行的174子集额外加进full结果。
- 目前validator沿用技术状态名 `PASS_SUBSET_COMPLETE`；full spec最终应结合manifest和1087/56完整coverage解释，不能仅凭该状态名断言论文复现。

本轮只实际运行了spec生成器及本地文件/参数检查，三种选择均成功生成；没有调用managed owner或申请GPU。
