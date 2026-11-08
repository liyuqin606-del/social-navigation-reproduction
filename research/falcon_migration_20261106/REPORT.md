# Falcon迁移第一步：真实接口已验证，世界模型尚未部署

2026-11-06（Asia/Shanghai）。使用既有官方Falcon检查点与Habitat环境，仅做工程探针。

## 实际完成

TACC作业28283，单张NVIDIA GeForce RTX3090，节点10-0-1-11。Slurm分配02:06:23–02:07:59 HKT，96秒；COMPLETED / 0:0，作业已离队、资源释放，managed owner为OUTPUTS_VERIFIED。申请上限20分钟，未占满。

三组各16步，共48个真实仿真步：

1. reference：冻结原策略，插入只读记录。
2. shadow：经过输入白名单适配器，仍执行原策略动作。
3. override：全局第3步预定覆盖为不同于原建议的左/右转，第6步预定执行STOP；其余仍执行原策略。

三组使用相同种子、同一已有Demo清单，并在独立进程启动。动作覆盖是工程故障注入，不是学习到的导航决策。

## 验收证据

| 检查 | 观测结果 |
|---|---|
| 合法输入 | robot jaw depth [1,256,256,1]、PointGoal [1,2] |
| Falcon头部 | 特征[16,512]，logits[16,4] |
| reference与shadow | 深度摘要哈希、目标、动作、回合键、reset与循环输入哈希逐步相同；特征/logits最大绝对差均0 |
| 实际动作回写 | 非reset步共44次previous_action检查通过；写入actions与take_actions的机器人标量切片后，经官方update_hidden_state更新，再核对env.step实发值 |
| 其他agent | 每步在修改前后核验其stored/executed动作切片相等 |
| STOP与reset | override第6步STOP得到done=True，第7步not_done mask正确表示reset；总reset计数由1变为2 |
| 权重与buffer | 三组actor state_dict前后SHA相同，无优化器、无参数更新 |
| 回收 | 35个输出逐文件字节/SHA核验，共505,018字节；原回执和输出保留 |
| 独立本地复核 | 在硬链接输入的临时目录重跑跨组验证，不改写已回收输出；PASS_REAL_FALCON_INTERFACE_ONLY |

跨组等价只覆盖这16步，不证明所有场景与所有执行配置完全等价。reset测试核验了官方mask链路，尚未验证未来新世界模型/option内部缓存的清理。

## 方法改进的结果与迁移决策

并行完成了一轮新的二维实验：可中断子目标、4步低层规划及刷新时首动作一致性。在新40个场景上，360回合结果为新方案SR87.5%、原接管95%、每步12步重规划90%。新增规则未通过收益门槛，负结果详见相邻研究目录`adaptive_option_20261106/REPORT.md`。

因此本次**未把新规则或二维模型部署到Falcon**。本次已验证的是接入路径：允许观测→真实特征/logits→机器人动作覆盖→实际previous_action→终止/reset；不是导航性能迁移。

## 数据和权限边界

沿用已经用于Demo的3个指定val episodes，仅作16步接口探针；没有拟合、校准、调参、排名或报告它们的导航成绩。探针特征被标记为ENGINEERING_ONLY，训练样本数为0，不允许将它们混入将来的训练数据。原1087评测剩余616条仍暂停，旧队列/冻结文件未改。

适配器白名单仅接收`agent_0_articulated_agent_jaw_depth`和`agent_0_pointgoal_with_gps_compass`以及真实Falcon头部输出。官方多agent观测batch中可能存在其他agent的特权观测，因此在构造frame之前明确剔除；没有将人体真值、全局地图、未来状态传给适配器。

二维模型使用完整地图、精确行人跟踪和已知运动通道，不能直接解释为Falcon深度输入下可用的世界模型。需要独立train场景和训练转移；最近对两个既有数据根目录的检查未发现train，尚未穷尽整个账号。

## 仍待完成的迁移环节

1. 准备原始train场景/episode，并冻结fit、开发、校准、确认的场景划分；不以现有val补空。
2. 基于已验证接口新增真实转移采集：当前合法观测/特征、实际动作、下一状态、真实terminal、截断类型。done后自动reset的观测不能当作同回合next_state。
3. 先用Falcon特征＋目标＋动作预测短时事件/价值，与同输入直接评价头比较；二维分类表不具有可直接迁移的参数意义。若引入行人跟踪或地图，必须从允许观测估计并重新记录输入协议。
4. 验证多步模型与候选的支持覆盖，再开展小型闭环开发实验。保留无模型规划、单步评价、每步重规划、固定接管等对照。失败的adaptive规则不作为默认部署候选。
5. 风险成本目前无真实覆盖证明；不要把旁路输入检查、STOP语义测试或仿真无碰撞称为物理安全认证。

## 工程检查的限制

本地3项bridge测试及8项managed测试通过。复制的旧fake-worker集成测试在macOS的`/proc/meminfo`读取处失败；复查确认为Linux平台前提，不计为通过，原worker未修改。真正Linux作业内的GPU身份、存储监测、运行、验收与释放随后均通过。它也不能补作历史scancel端到端清理测试。

远端日志有既有语义场景描述加载警告；本次只使用深度/目标输入，三组执行与接口验收完成，但不据此声称语义感知资产完整。全部日志已回收保留。

主要文件：`bridge.py`、`derive.py`、`run_case.py`、`validate_probe.py`、`LOCAL_VALIDATION.json`、`plan/run/DRIVER_RECEIPT.json`、`plan/run/SCHEDULER_LATEST.json`。
