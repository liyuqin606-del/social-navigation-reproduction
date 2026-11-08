# Falcon 官方 baseline 复现准备

核查：2026-11-03。状态：源码和配置已核对，尚未加载权重、安装仿真环境、执行模型或提交GPU。代码位于 `/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon`。未发现工作区父目录或该仓库中的AGENTS.md。

## 代码身份

官方仓库 `https://github.com/Zeying-Gong/Falcon`，固定提交 `d15ef512c4b1451f9daeaf6d4d7ecc702ed57430`。本地浅克隆成功，738个已追踪文件，共6,343,594字节；全文件字节和SHA记录在同目录 `falcon_source_manifest.json`。本次未修改上游代码。

逐项输入清单见 `falcon_input_requirements.json`。未知量保留null，没有用估计值冒充实际盘点。

## 最小模型输入

官方Drive目录的元数据与summary.txt已可读取。summary说明：`falcon_noaux_25.pth`是完整Falcon去掉训练辅助模块后的推理版，不是论文中的“未使用辅助任务训练”消融模型。

|文件|Drive ID|字节|用途|
|---|---|---:|---|
|falcon_noaux_25.pth|1t9DGtwjAdPu7yDPBC8sHTGdXAnbai4tM|50,048,730|正式推理权重|
|pretrained_habitat3.pth|1QakOj1cUVPoEmTe5_aNwyMyDsBYODAyX|49,961,178|原样eval配置初始化时也会加载|
|falcon_pretrained_25.pth|1uG3O2BMwrl7ZWHh3j79zc55vbGBYAjyz|138,656,437|用于继续训练，首轮评测不需要|

预期放在仓库根目录的`pretrained_model/`。上述字节为Drive元数据，尚无实际下载文件SHA。

第二个文件不能因为“最终会被Falcon权重覆盖”就忽略：官方eval YAML设`ddppo.pretrained=True`，`single_agent_access_mgr.py:201–209`在构建策略时直接加载基础模型，没有eval豁免；随后`ppo_trainer.py:878–883`才加载正式检查点。若改为`pretrained=False`须明确记录覆写；原样入口建议准备两个文件。

权重来源：[官方目录](https://drive.google.com/drive/folders/1Bx1L9U345P_9pUfADk3Tnj7uK01EpxZY?usp=sharing)、[模型说明](https://drive.google.com/file/d/1U3IdPQeG6vV7XzbU8iH13A-DpLloRY3f/view)。

## 数据与访问条件

官方episodes目录提供完整zip，而不是已确认的独立val包：

|文件|Drive ID|压缩字节|eval所需路径|
|---|---|---:|---|
|social-hm3d.zip|1c0KBYaj8FUzUQl-lGG6IjfyC4KeMZJcZ|2,409,516,674|`data/datasets/pointnav/social-hm3d/val/val.json.gz`以及`val/content/*.json.gz`|
|social-mp3d.zip|16KGr9cae1z3ypfgeeDSd5wDnJTfu2HhK|2,106,409,739|`data/datasets/pointnav/social-mp3d/val/val.json.gz`以及`val/content/*.json.gz`|

数据yaml本身的`split: train`不代表评测使用训练集：`ppo_trainer.py:842`在eval时改成`habitat_baselines.eval.split`，默认是`val`。应保留官方val，而不能直接把yaml里的train当目标。

目前未读取zip中央目录或gzip内容，因此完整val的episode数、场景列表及解压字节还未知；下载/流式提取后必须清点。README声明数据集总体844个HM3D场景和72个MP3D场景，不能用这些总体数代替val的实际数量。只运行部分val必须标记为子集评测。

Episode文件不包含场景网格。还必须准备它们引用的HM3D/MP3D meshes、scene dataset配置及相关资产。HM3D官方获取流程需要已获数据访问权限及Matterport API token；MP3D通过官方授权的download_mp.py获取Habitat archive。若账号或远端已有获准数据，可复用并核验；本任务没有注册账号、接受新条款、寻找第三方绕过镜像或使用他人凭据。[Habitat官方数据文档](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)。

官方Habitat-Lab资料给出的整个Habitat格式HM3D archive约130GB，MP3D约15GB；它们不是本任务val峰值估算，不能据此直接提交。最小完整评测应先解析val中的scene_id，再匹配所有必要场景；空间预算需把压缩文件、解压文件、环境和临时缓存同时计入。

Habitat公共test scenes、单个MP3D example、HM3D minival能用于环境检查，但改变了scene/episode集合，不能称为完整Falcon论文评测。

## 仿真资产与环境

README要求下载UID：`hab3-episodes habitat_humanoids hab3_bench_assets hab_spot_arm`。按实际配置至少有：

- `data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf`及引用资产。
- `data/robots/spot_data/spot_walking_trajectory.csv`，来自[官方Habitat附件](https://github.com/facebookresearch/habitat-lab/files/12502177/spot_walking_trajectory.csv)。
- `data/humanoids/humanoid_data/{id}/{id}.urdf`和`{id}_motion_data_smplx.pkl`；id为female_0、female_1、male_0、male_1、female_2、female_3，另需URDF引用的网格/纹理。
- 仿真配置与episodes实际引用的physics/scene/assets文件。

官方安装：Python3.9、Habitat-Sim0.3.1 withbullet/headless Linux、editable habitat-lab及habitat-baselines、根requirements。根requirements固定numpy1.23.5与moviepy1.0.3；其余依赖大多没有完整锁定，尤其torch只有`>=1.3.1`，所以不能把现代默认pip解出的环境说成原作者环境。首个实际运行后需保存conda explicit和pip freeze。

README原始入口为`python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d.yaml`（另有falcon_mp3d.yaml）。安装后等价模块入口`python -u -m habitat_baselines.run ...`须在repo根目录执行，使`falcon`和相对data路径可解析；环境导入应先验证。

## 不能默改的评测协议

当前官方配置/默认值：seed100；8并行环境；500步上限；完整`test_episode_count=-1`；每episode评测一次；`use_ckpt_config=False`；深度256×256，范围0–10m并归一化；深度+GPS/Compass点目标；512维两层LSTM；四动作STOP/前进/左右转。`FALCONEvaluator`显式`deterministic=False`，即采样动作，不能无说明切换argmax后比较论文数字。

STOP会结束episode。HumanCollision指标检测到人体碰撞后也令`task.should_end=True`。PSC阈值为1.0m，默认测地距离，找不到路径时退回欧氏距离；成功距离默认0.2m并要求STOP。这些均属于当前代码协议，应记录而不是调整到“看起来更好”。

当前官方`socialnav_task_detail.yaml`不启用STL，虽代码含STL类，默认`output/result.json`只包含SR/SPL/PSC/H-Coll及自定义Total。不能声称默认输出已复现论文全部五项指标。若增加纯日志/指标导出，需保存补丁、证明不改策略/终止规则，并区别默认评测与增强审计版本。

`output/result.json`和`output/actions.json`都是固定相对路径；各run必须隔离工作目录或立即封存，防止第二次运行覆盖第一份。原evaluator只把逐episode指标保留在内存后聚合，actions.json不是逐episode指标证据；需要独立记录时应加纯记录instrumentation。

## 论文目标与复现范围

用户提供ICRA2025正文表II（PDF第6页）报告三次不同随机种子训练/运行的均值±标准差：

|方法/数据|Suc|SPL|STL|PSC|H-Coll|
|---|---:|---:|---:|---:|---:|
|Falcon/Social-HM3D|55.15±0.6|55.15±0.7|54.94±0.7|89.56±1.4|42.96±1.1|
|Falcon/Social-MP3D|55.05±0.7|55.04±0.6|54.80±1.0|90.01±1.2|42.19±0.9|

全部为百分比。NavThinker表I的Falcon56.26/52.05等是另一文复评口径，不应混成当前Falcon原文目标。公开单个检查点的eval可以复现发布模型表现，但即使用三个eval seed也不等于三个独立训练seed。

若后续需要从头复现训练，原文是HM3D train、PointNav初始化、10M steps、4×3090、8并行环境、3个seed；README训练YAML含预算字段与原文有出入，必须先比对总步数和全局batch，不能在1–2卡上随意缩短后宣布完整训练复现。当前任务先准备已发布检查点评测。

下一实质关口：核验可访问scene数据及完整val清单 → 建好环境/输入SHA → 原样官方入口短运行确认 → 完整val评测与结果回收。只有这些输入齐备才值得申请GPU。
