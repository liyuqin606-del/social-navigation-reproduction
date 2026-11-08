# Falcon 外挂模块与权重编辑入口（只读核查）

结论：两条路线都能实现。优先尝试四动作头或在其 logits 上附加小型风险预测器；是否提升必须通过新的、独立的闭环评测确认。此次未改权重、未申请 GPU，42 个当前队列绑定输入 SHA 未变。

检查点为作者发布的 `falcon_noaux_25.pth`，SHA-256 `ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca`。使用已有受限元数据读取器，仅解析 inert tensor shape，不导入上游类、不重建张量、不运行模型。完整形状记录见同名 JSON。

| checkpoint[0]["state_dict"] 中的精确键 | 形状 | 作用 |
|---|---|---|
| `action_distribution.linear.weight` | 4 × 512 | 四动作 logits 的线性权重 |
| `action_distribution.linear.bias` | 4 | 四动作 bias；与上一项共 **2052 个参数** |
| `net.visual_fc.1.weight` / `.bias` | 512 × 2048 / 512 | 深度视觉特征投影 |
| `net.tgt_embeding.weight` / `.bias` | 32 × 3 / 32 | 目标距离和方向嵌入；键名确实拼作 embeding |
| `net.prev_action_embedding.weight` | 5 × 32 | 四动作加 episode 开始 token |
| `net.state_encoder.rnn.weight_ih_l0` | 2048 × 576 | 第一层 LSTM 输入权重 |
| `net.state_encoder.rnn.weight_hh_l0` | 2048 × 512 | 第一层 LSTM recurrent 权重 |
| `net.state_encoder.rnn.weight_ih_l1` / `weight_hh_l1` | 各 2048 × 512 | 第二层 LSTM |
| `critic.fc.weight` / `.bias` | 1 × 512 / 1 | 状态价值，未进入动作采样 |

当前真实输入为 **jaw depth + pointgoal/GPS compass**，没有 RGB、语言指令或 oracle 行人未来轨迹。视觉 512 维、目标 32 维、上一动作 32 维拼成 576 维，两层 LSTM 输出 512 维特征。官方配置的 RGB 注释未启用；检查点第一层卷积也是单通道 `[32,1,7,7]`。依据：[官方评测输入](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d.yaml:32)、[视觉与 LSTM 构建](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py:578)、[前一动作与 LSTM forward](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py:746)。

推理路径是 `net → 512维 h → Linear(512,4) → Categorical → sample`。官方 evaluator 显式 `deterministic=False`。critic 虽计算 value，却没有参与动作选择；修改它不会改变当前推理动作。`act()` 不调用辅助 loss；179 个实际 state_dict 键没有 aux/human/future/people 头，实际评测的 auxiliary_losses 为空。不要把训练时期辅助损失存在，误解成发布推理模型自带可外挂调用的轨迹预测器。依据：[NetPolicy.act](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ppo/policy.py:324)、[CategoricalNet](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/utils/common.py:85)、[辅助 loss 仅在 evaluate_actions 中计算](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ppo/policy.py:391)、[随机评测入口](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py:142)。

动作索引固定为 **0=STOP、1=前进、2=左转、3=右转**。STOP 设置 `is_stop_called=True` 和 `should_end=True`，会结束本回合；成功还要求距离目标小于 0.2m。它不能充当避让时的临时等待。仓库有独立 Pause 实现，但当前四动作空间没有它；新增等待会改变动作空间和检查点接口。依据：[索引映射](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-lab/habitat/gym/gym_wrapper.py:180)、[STOP](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/falcon/additional_action.py:51)、[独立 Pause](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/falcon/additional_action.py:58)、[成功定义](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-lab/habitat/tasks/nav/nav.py:529)。

建议先做保留当前观测和四动作的外挂：使用深度历史或 h，对各动作预测未来碰撞/侵入个人空间的风险，再在采样前修改 logits。最稳妥的接口是 robot agent_0 的 `action_distribution`；只改最终 `envs.step` 动作会让 `prev_actions` 仍记旧动作，造成记忆输入与执行不一致。若替换动作，必须同步 action history、log probabilities 和 episode reset/pause 索引；在分布采样前改 logits 可避免这类双轨状态。依据：[更新 prev_actions 早于 envs.step](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py:151)、[env_actions 与 take_actions](/Users/yuqinli/Documents/ChatGPT/VLN/baselines/repos/Falcon/habitat-baselines/habitat_baselines/rl/ppo/policy.py:87)。

直接权重编辑可以从 `W,b` 的受约束增量开始，例如 rank-1 的 ΔW 或目标动作行。但若外挂是线性校正 `Δz=ΔWh+Δb`，它与 `W←W+ΔW, b←b+Δb` **在代数上完全等价**；不能把安装位置差异作为创新。非线性、动作条件的未来风险模型才可能提供简单静态头编辑没有的信息，但需要训练数据和真实预测/控制证据。RGB/VLM 外挂增加当前未使用的感知输入，应单独报告观测与推理开销，不能混同为同输入权重编辑。

当前 NumPy editor 与约束/恒等性检查仅说明算子可实现，不构成 Falcon 闭环收益或 SOTA 证据。当前完整 val 正用于 baseline 评测；后续编辑拟合与超参选择应使用独立开发数据，保持比较时的观测、动作、采样协议和测试集一致。
