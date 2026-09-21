# 完整 Topomap Adaptive：外部观测与仿真适配

## 当前状态

**三段适配代码已实现并装配，尚未运行验证。** 本轮按用户要求没有安装环境、
加载模型、请求 VLM、启动仿真或执行测试；只做代码检查和 Python AST 语法检查。
不能将这一状态描述为“完整版已跑通”或“Go2 已部署”。

早期 diagnostic 仿真是独立的红色标记回归工具。它过去的测试、视频和四个通过
场景不证明这里的完整版正确。完整路径不调用 MarkerDetector、简化 Navigator
或 DepthGridMapper。

配置仍是用户指定的 **Adaptive-hybrid**：deploy/full/adaptive_mac.yaml 继承
cfg/eval_goatbench_hgr_dual_topo_adaptive_hybrid.yaml 及其原版继承链。
策略依据：[Adaptive 实现说明](HGR_ADAPTIVE_HYBRID_IMPLEMENTATION.md)。

原评测入口 run_goatbench_evaluation.py 和 HypothesisAwareNavigator 的策略实现
没有修改。本轮修改原 src 的范围是外部 Scene 构造、共享感知初始化、显式类别表，
以及将 Habitat 专用导入移到实际需要它们的函数内。

## 已接好的流程

    MujocoPorts：实转扫描，同步 RGB-D / 内参 / 拍摄位姿
      ↓
    Scene.from_external → 原 YOLO-World / SAM / CLIP / ConceptGraph
      ↓
    原 TSDF.integrate → 原 Snapshot 聚类与观察历史
      ↓
    原 Frontier / HypothesisNodePredictor / HGR
      ↓
    OriginalAdaptiveLifecycle
      原 Place / PersistentBeliefTopology / scene_map / goal_graph
      sync → continue_choice 或原 navigator.select → setup
      ↓
    原 KnownSpaceExecutor.prepare / RouteGuidance.step
      输出一个算法级运动段，不执行 TSDF.agent_step
      ↓
    MujocoPorts：证书/新鲜度/机身通行检查 → LightNav MPC → mj_step
      ↓
    新帧、实测 XY / 朝向 / 停止速度
      ↓
    原 record_motion_progress / recover / feedback
      → Frontier 的 SemanticCritic 与级联纠错
      → 历史实例目标的 verify_entity / 新视角 / 停止
      → 下一轮观测

| 代码 | 已实现内容 |
| --- | --- |
| src/scene_goatbench.py | Scene.from_external，与原 Habitat 构造共享感知初始化，不创建 simulator/navmesh |
| src/coordinates.py、src/utils.py、TSDF 和查询模块 | 纯坐标转换与延迟 Habitat 导入 |
| topomap_deploy/full/observations.py | 当前周期真实帧缓存；按采集位置、朝向、实际视场匹配前沿画面 |
| topomap_deploy/full/original.py | 原感知、TSDF、Snapshot、前沿和验证调用；坐标适配、反馈融合、降级检测 |
| topomap_deploy/full/lifecycle.py | 具体 OriginalAdaptiveLifecycle；意图延续、计步、恢复、来源复核 |
| topomap_deploy/full/safety.py | 不可变已知空间地图快照、地图证书和运动端通行检查 |
| topomap_deploy/full/sim_io.py | 传感器/运动端；独立控制线程和 watchdog；实转扫描、MPC、实测停止反馈 |
| topomap_deploy/full/bootstrap.py | 本地模型加载、原 Scene/TSDF/拓扑构造、完整仿真装配和关闭清理 |
| tests/test_full_topomap_framework.py、tests/test_full_topomap_adapters.py | 状态机和适配器测试代码；本轮未执行 |

AdaptiveLifecycle Protocol 保留作接口契约，现在有具体实现，不再只有待接接口。
完整装配使用同一个原 HypothesisAwareNavigator(stage="adaptive")。

## 保留的 Adaptive 行为

- 跨 goal 保留 Scene、对象别名、Snapshot、Place 和 scene_map；只重置 goal/intent。
- 先调用原 continue_choice，释放后才重新请求原 HGR；不另写目标选择器。
- Frontier 的三步窗口统计**实际发生移动的算法级运动段**，不是 MPC tick。
  扫描旋转不占 Frontier motion_steps；运动反馈按 route ID 去重。
- setup、KnownSpaceExecutor、RouteGuidance 只生成请求。原 TSDF.agent_step
  会提前标记建议位置已探索并清空终点，因此外部执行不调用它；
  相同的周围已探索更新延迟到实测反馈后，作用于真实位置。
- 历史来源、image/description、REVISIT 三个条件由原任务 requires_verification
  决定。confirmed 经原 feedback 接受后才停止；uncertain 使用原
  RouteResolver.object_view 排除已检查位置后申请另一个视角；
  rejected/error/无合法新视角则释放重选。
- 普通 Snapshot 到达沿用原停止规则；Frontier 到达不是目标完成。
  Frontier 的新帧经过真实感知后，交给原 SemanticCritic 与级联删除逻辑。
- 原 predictor 回退到 heuristic，或 Critic 出现 feature residual 错误时，
  完整路径明确失败停车，不把降级结果记作完整 VLM 成功。

## 观测、坐标和执行边界

外部契约是 map z-up、base x前/y左/z上、camera optical x右/y下/z前。
TSDF 使用标准光学位姿，ConceptGraph 使用原 Habitat/GL 位姿。
原 ConceptGraph 的纵向像素索引为 H-1-v，因此仅在其入口将 cy 转为 H-1-cy，
避免额外像素偏差；TSDF 内参不变。公式测试已写，仍需实际渲染器/相机校准。
深度必须以米为单位、与 RGB 配准。

前沿取图只查已采集画面，不移动机器人或修改 qpos。当前 MuJoCo 每轮做 12 个
朝向的完整实转扫描并回到原主朝向。每帧记录实际位姿；扫描偏移超限、视场不足
或缺少前沿朝向时停车报错。与原 Habitat runner 的瞬时多视角相比，物理扫描的
采样数量、耗时和图像不同，需要单独验收。

地图范围是配置提供的固定 TSDF 体积，不读取完整仿真地图、隐藏目标或 navmesh。
沿用原初始化局部清空先验 init_clearance * 2，不能声称初始盲区已被传感器测量。
机器人离开体积或 map epoch 改变时拒绝继续；自动扩图和 SLAM 重定位恢复不在
当前适配中，需要重建一致会话。

运动端复核地图 revision/hash、路线起点、新鲜度和机身通行范围；独立处理深度
停车、地图变化、限速、超时。反馈在实测线速度和角速度均降到阈值后产生，并
检查刹车后的 XY/朝向。规划/模型调用期间控制线程持续发零速。

watchdog 独立于模型主线程；MPC 超时后的旧命令会失去执行权限。若整个物理线程
阻塞，仿真时间也暂停，恢复时不执行过期非零命令。这是仿真约束，
**不能作为 Go2 的硬件独立急停或安全认证**。

## 装配入口与默认关闭

--describe 只检查配置，不启动模型或仿真。本轮没有执行以下命令：

    cd /Users/agiuser/Documents/code/topomap
    .venv-sim/bin/python -m topomap_deploy.full --describe

程序化入口 bootstrap.start_simulation 返回 SimulationSession(workflow, core, ports)。
通过 workflow.begin(Goal(...)) 和调用方事件循环中的 workflow.advance() 推进，
退出时 session.close()。尚未增加 CLI --run，也没有 ROS/真机启动入口。

程序化执行同时要求：

1. 调用方明确将 full_framework.execution_enabled 设为 true；仓库默认 false。
2. 调用 start_simulation 时明确传 allow_unvalidated=True，承认适配尚未验收。
3. 本地依赖、模型文件和配置所指的 VLM 环境变量已准备。

只构造核心可调用 assemble_core(cfg, initial_frame=..., models=..., root=...)。
将来 Go2 复用该核心，但需要另行实现真实机器人 SensorPort/MotionPort。
这些函数本轮仅写入代码，未调用。

## 后续仍需准备和验证

1. **环境与模型**：此前 .venv-hgr-mac 安装中断，不能视为可用环境。
   原 YOLO-World、SAM、OpenCLIP、Open3D、FAISS、HiPart 等依赖及 Mac 兼容性
   尚未验证。需本地 YOLO、SAM、ConceptGraph 的 LAION ViT-B-32 权重，以及
   YOLO-World 自己的 OpenAI CLIP ViT-B/32 文本编码器权重。后者需要正确的
   上游 clip 包，不能误装同名无关包。加载器只接受存在的本地文件，不负责安装
   或下载；YOLO 文本编码器预载避免 set_classes 隐式下载。
2. **语义仿真场景**：scene_xml_path=null 使用早期墙体/方块 fixture，它没有
   足以检验 YOLO 导航的真实语义资产。可指定外部 XML，须保留当前 planar proxy
   的三个自由度、执行器和 RGB-D 相机契约。需要包含可识别椅子、桌子等资产的
   室内场景；不能用红色检测替代来宣称通过。
3. **运行测试**：执行新增接口测试；验证禁止 Habitat 导入时外部 Scene 可加载；
   校准 RGB-D/地面高度/坐标，检查真实模型与 VLM 失败路径。新增测试尚无通过记录。
4. **算法对齐**：同输入回放对比原 runner 的对象身份、Snapshot、提示词与映射、
   选择、来源绑定、路线和停止事件。代码复用不等于已经证明行为等价。
5. **完整闭环验收**：验证直达、绕障、跨 goal 记忆、三步 Frontier 窗口、
   uncertain 新视角、堵塞恢复、掉帧/超时/取消。之后才接 Go2 四足模型、
   ROS/真实定位及真机。当前仍是仿真理想位姿，不是 SLAM。

本轮交付是三段具体适配实现与装配代码，不是可直接用于实机的部署包。
