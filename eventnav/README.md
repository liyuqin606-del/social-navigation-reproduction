# EventNav：忠实于理论的有限参考求解器

实现依据：`research/sota_temporal_20261105/THEORY_ADVANCED.md`（v0.3）。目前是模型已给定的CPU参考实现，适用于有限信息状态、有限时域、有限事件核顶点和有限后继预算网格。没有接入Falcon环境，没有训练模型，没有新的导航成绩。

## 理论与代码逐项对应

| 理论对象/公式 | 实现 | 明确限制 |
|---|---|---|
| 状态(h,r,n,t) | `Planner.solve(state,budget,interventions_left,time_left)` | state必须由调用方提供充分的信息状态；字符串不等于已经证明充分的神经表示 |
| K_o(ℓ,e,h') | `Outcome` + `Option.kernels` | 核顶点的凸包；完整概率和严格为1 |
| 互斥C/G/F/R | `Outcome.event` | 联合碰撞/成功标签尚不支持，接入前必须先核对实际官方事件规则 |
| 正执行时间 | duration>=1，且不超过当前剩余时域 | 不会静默截断长option；截止时刻以后无未来收益/风险 |
| 方程(1) | max_K [P(C)+Σ_R p·child.budget] <= budget | 使用后继条件预算；每个正概率R分支必须有可行后继策略 |
| 方程(2) | min_K [P(G)+Σ_R p·child.utility]−κ·I | 干预成本每次收取一次；尚无额外时长成本 |
| 切换边界 | 比较所有可行continue/intervene选项，返回最大值 | 相等时优先低风险界与不干预；没有假定标量风险阈值 |
| 次数预算 | 每个干预option消耗一次n | 一个option内部可执行多步，仍计一次高层干预 |
| 矩形歧义集合 | `Model.rectangular` 入场条件 | 非矩形模型直接拒绝；声明为True不是数据驱动的覆盖证明 |
| 策略承诺与预算传递 | `PolicyCursor.observe` | 后继策略与预算由已观察到的duration/state决定，不能按不可观察的核编号选择 |
| 无可行解 | `Infeasible` | 只说明当前确定性option与预算网格内不可行，不等于连续预算/随机策略全局不可行 |
| 预算/状态信息不足 | ValueError / SearchLimit | 不把未知或搜索截断伪装成不可行或安全回退 |

所有概率、成本和预算使用Fraction精确计算。输入浮点数被拒绝，调用者须提供如`"0.1"`的十进制字符串。没有用于放松风险约束的浮点容差。

`audit.py` 独立遍历输出策略树，重新计算真实模型下的最坏碰撞概率、最差成功概率与效用，以及最大干预次数；它不使用缓存的value作为递推值，并校验预算证书。检查使用显式异常，Python优化模式不会去掉检查。

## 最优性与安全性的范围

求解器穷举后继预算分配，递归求解可行后继策略。对给定网格与确定性option策略类，采用矩形多面体核的逐步max/min，得到该受限类的Bellman最优值。因概率函数线性，凸包上的极值在提供的顶点达到。没有实现随机option混合、连续预算优化或一般POMDP求解。

这是有意保留的可审查参考版本，时间复杂度会随handoff分支数和预算网格指数增长。SearchLimit表示尚未完成搜索。后续快速求解器必须与本实现及独立枚举基准比对，不能只凭输出相近就宣称等价。

结果统一标记 `CONDITIONAL_ON_SUPPLIED_RECTANGULAR_MODEL`。即便计算完全正确，也没有证明学习模型包含真实环境。未知人体、观测混叠、分布偏移、神经置信区间都尚未解决。传入“rectangular=True”不赋予物理安全认证。

只优化风险约束下成功效用，未强制相对于Falcon的SR非劣约束；极严预算下可能选择失败STOP。v0.2的四动作相对优势LP是另一目标，尚未在本包实现，不能与本求解器混称。必须报告成功—风险曲线。当前包也不实现SPL/PSC保证、几何可达管或连续时间认证。

## 运行

在项目根目录：

```sh
PYTHONDONTWRITEBYTECODE=1 baselines/repair_20261104_e/runtime/bin/python3.12 -B -m unittest discover -s tests/eventnav -v
PYTHONDONTWRITEBYTECODE=1 baselines/repair_20261104_e/runtime/bin/python3.12 -B -m eventnav.demo
```

示例是合成的两阶段事件树：风险预算10%时选择继续原策略，碰撞上界9.84%、成功下界90.16%；预算9%时选择一次绕行，碰撞上界5%、成功下界80%。这些数字仅用于验证预算如何改变最优切换，不是Falcon实验指标。

## 接入实机/仿真之前的必需环节

1. 为option提供明确的原始动作执行器、终止规则和事件优先级。
2. 验证完整时钟/重置边界与实际executed previous_action；不能把STOP变成WAIT。
3. 训练或构造包含handoff后继信息的事件核；现有竞争风险头没有此输出，不能直接冒充此核。
4. 说明信息状态充分性、核集合构造与联合覆盖证据；经验模型必须继续保留条件性标签。
5. 实际执行已选option后，用观测到的outcome推进PolicyCursor。遇到支持集外事件将使证书失效；外部执行器的停止/回退行为须另设计，不能把抛异常当作物理安全制动。

所有代码位于新目录，未修改既有冻结评测入口，未提交GPU或恢复暂停任务。
