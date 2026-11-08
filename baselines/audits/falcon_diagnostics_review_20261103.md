# Falcon / combined diagnostics 独立入口审阅

日期：2026-11-03。只读源码、checkpoint ZIP 中的 pickle opcode；没有反序列化 checkpoint、导入模型或调用 CUDA。对包装脚本的修复已经 root 明确授权。

## 已确认并修复的两个问题

1. **官方 checkpoint 顶层布局读错。** `falcon_gpu_smoke.py` 原先读 `ckpt['state_dict']`。本地官方 `falcon_noaux_25.pth` 的 `data.pkl` 开头为 `EMPTY_DICT → BININT1 0 → EMPTY_DICT → 'state_dict'`，所以实际是 `ckpt[0]['state_dict']`。这与官方 `MultiAgentAccessMgr.load_state_dict` 先尝试 `state[agent_i]` 一致。修复只使用已确认的整数0布局，不加通用fallback；内部 `net.*` 参数键不改，仍严格加载。结果及validator新增 `checkpoint_layout='multi_agent_integer_0/state_dict'`。`torch.load` 显式 `weights_only=False`，和原作者加载方式一致，且先核验固定SHA。

2. **Falcon fork导入依赖启动cwd/环境。** 远端准备只将 `habitat-lab`、`habitat-baselines` editable安装，没有安装repo根目录的 `falcon`。早先 `python -c` 在repo cwd导入通过，不能保证运行外部 `prep_scripts/falcon_gpu_smoke.py` 时找到它。新增 `--repo`，在任何Habitat导入前显式置顶repo、habitat-lab、habitat-baselines三条源码路径，随后检查实际 `falcon/habitat/habitat_baselines.__file__` 均位于该repo。combined runner传入 `root/src/Falcon`，结果及validator保存/复核真实模块路径，避免误导入NavThinker同名包。

## 已核对的一致性

- 自定义配置注册由 `falcon/__init__.py` 导入 additional_action、additional_sensor、additional_metric、auxiliary_tasks；这些模块包含对应ConfigStore注册。smoke现显式导入 `habitat_baselines.run`，与官方入口一致。`get_config` 自身也调用 `patch_config`。
- 直接策略 smoke 使用 agent前缀移除后的 depth/pointgoal空间与4离散动作；LSTM状态的官方属性为4×512，batch2后为2×4×512；PolicyActionData字段和mask维度匹配。
- smoke不调用SingleAgentAccessMgr初始化权重路径，因此只需要发布Falcon checkpoint。正式subset eval仍需要两个官方checkpoint，runner已分别记录。
- `diag/falcon` 的smoke输出与 `eval/` 的正式子集输出没有重名冲突；正式runner自行把 `VLN_RUN_OUTPUT` 设为传入eval目录，其派生evaluator不会写原repo固定 `output/`。
- subset runner的 `scene_id` 映射与官方PointNav loader一致；filesystem alias路径保留，manifest中的原始episode未改；4场景73及9场景174两套均已精确核对。
- 独立subset validator依据逐回合原始标量和精确keys检查，而smoke validator只认可合成观测/空场景、benchmark count0。两类证据未混用。

## 已修复的超时/进度记录

combined runner原来的 `subprocess.run(..., timeout=240)` 未捕获 `TimeoutExpired`，某阶段超时会跳过剩余诊断和summary。现按root授权捕获超时为exit124、启动错误为exit127，保存每阶段开始/结束状态并继续其余诊断；summary使用临时文件替换。只有全部六阶段到达结束、Falcon smoke成功、三个独立validator均成功且无超时，最终才成功。独立combined validator再次检查阶段集合和超时标记，超时不会因为留下旧JSON而被接受。新增非空输出拒绝，防止复用旧诊断结果。

审阅后的四个diagnostic脚本通过Python AST检查；没有新增模拟结果测试，没有把静态审阅写成CUDA运行通过。最终以Slurm日志、真实smoke JSON及子集validation为准。
