# 本机 MuJoCo 导航闭环

这里是早期**诊断标记闭环**，不是完整 topomap 算法。完整版的独立框架及当前
已实现适配器及待验证项见 [完整 topomap 框架](../../docs/FULL_TOPOMAP_FRAMEWORK.md)。

这一阶段在 Apple Silicon Mac 上验证导航工程链路：

```text
MuJoCo 渲染 RGB-D → 深度建图 → topomap Navigator/HGR
       ↑                            ↓ Observation v1 / 本机 HTTP
    mj_step ← 有限力执行器 ← LightNav 的真实 MPC 求解器
```

机器人位置由 MuJoCo 动力学推进、反馈回来；运行中不直接修改机器人 `qpos` 来移动。
有物理碰撞响应，可独立用测试验证“持续给前进命令也不能穿过障碍物”。

## 环境

当前实现使用 Python 3.11、MuJoCo 3.3.7、CasADi 3.x、PyTorch 2.5.1、NumPy 1.x。
MuJoCo/CasADi 在本机运行，PyTorch 只用于加载现有 HGR 代码，不加载导航模型权重。
不用 ROS 2、CUDA、Docker，也不需要模型 API key。

在 topomap 根目录安装独立环境：

```bash
bash scripts/setup_navigation_sim.sh
```

Python 和依赖分别放到 `.python-sim`、`.venv-sim`、`.cache-sim`；不修改系统 Python。
首次需下载约数百 MB 的依赖。不要在 macOS 设置 `MUJOCO_GL=egl`；这里使用默认
macOS 离屏图形后端。

## 运行

将下面路径换成实际的 LightNav checkout。只引用其中 `vln_mpc` 的求解器源码，
不启动它的机器人 launch，也不修改 LightNav 文件。

```bash
.venv-sim/bin/topomap-sim \
  --backend diagnostic \
  --lightnav-dir /Users/agiuser/Documents/code/LightNav-0 \
  --case all \
  --output artifacts/simulation/my-first-run
```

输出路径中的各用例目录必须不存在，避免覆盖旧结果。每个用例生成：

- `closed_loop.mp4`：第一视角、仿真俯视图、深度生成的地图与实际轨迹；运动部分
  约 2 倍速，判断时序请以画面仿真时间和报告为准。
- `final.png`：最终状态截图。
- `trace.json`：逐控制周期位姿、命令、门控、地图覆盖和评估距离。
- `report.json`：通过/失败、位移、接触次数、停止漂移和验证范围。

总结果保存在输出根目录的 `summary.json`。任一用例未满足条件，进程退出码为 1，
不会仅因“画面能打开”就报告成功。用 `--case direct` 单跑一个用例；
`--no-video` 可跳过视频编码，`--max-seconds` 设置每个导航阶段的仿真时长上限。

## 验收用例

| 用例 | 检查内容 |
| --- | --- |
| `direct` | 从图像检测标记并通过深度定位，实际移动到阈值范围，连续确认后停止 |
| `detour` | 中间有实体障碍，地图来自深度观测，规划并运动绕行到目标 |
| `sensor_loss` | 行驶中停止提供新图像/深度，在数据有效期到期后输出零速度 |
| `planner_loss` | 行驶中关闭本机 HTTP 服务，持续请求并记录连接失败，旧路径过期后输出零速度 |

导航通过要求：声明到达且机身原点到目标箱体表面的实际水平距离不超过 1.3 m、
实际移动超过 1.5 m、无机器人与环境接触、最后零命令后漂移小于 4 cm。
故障用例要求故障发生前已移动超过 0.3 m；传感器断流后 0.7 秒内、服务中断后
1.6 秒内开始持续输出零速度；无接触，最后平移和转动速度均收敛到近零。
以上时间均为仿真时间。仿真先通过实际原地旋转采集初始地图，不能直接查看
整场景导航网格。

辅助测试：

```bash
.venv-sim/bin/python -m pytest -q \
  tests/test_simulation.py tests/test_go2_deploy.py \
  tests/test_go2_bridge.py tests/test_go2_http.py
```

## 本机实测结果

`artifacts/simulation/validated-run-01/summary.json` 记录本次四个场景全部通过；
上述测试共 **51 项通过**，包括四个实际闭环回归用例。视频首尾帧已解码检查，
绕障最终图已核对实际轨迹。

| 场景 | 实测结果 |
| --- | --- |
| 直达 | 行驶 3.236 m，最终距目标表面 1.201 m，零命令后漂移 2.53 cm |
| 绕障 | 行驶 4.263 m，最终距目标表面 1.148 m，创建 5 个 HGR 假设节点 |
| 传感器断流 | 断流后 0.50 s 开始持续零命令，此后最终停稳；故障后共移动 0.124 m |
| 规划服务关闭 | 记录 8 次 HTTP 连接失败，1.40 s 后持续零命令；故障后共移动 0.302 m |

四个场景均无机器人与环境接触。故障用例“通过”表示按预期停止，不是到达目标；
这些是固定场景、理想传感器条件的工程回归，不是通用导航成功率或真机安全认证。

本次闭环还暴露并修复了一个实际规划问题：旧接近点可能落在到达阈值之外，造成
机器人停在目标附近但永远不返回到达。现在接近点上限位于到达阈值以内，并新增
对应回归测试；早期失败记录保留在 `artifacts/simulation/dev-direct-01/`。

## 这次验证的边界

`--backend diagnostic` 必须显式指定，不会自动把真实模型替换成测试输出：

- **感知**：目标是红色几何标记，用当前 RGB 图像的颜色连通域检测，不是 YOLO/VLM
  识别椅子等真实物体；检测器不读取仿真目标 ID 或目标位置。
- **建图**：用实际深度射线增量生成占据图；没有读取全场景障碍几何。假设平地，
  将机身中心半径 0.45 m 内未知格视为自由空间（近场平地先验，不清除已知障碍）；
  静态障碍证据保守保留。这不是 RTAB-Map 验证。
- **定位**：使用理想仿真位姿，不验证 SLAM 漂移、回环或传感器标定误差。
- **导航**：使用真实 `Navigator`、HGR 图结构与启发式 predictor、已知空间 A*、
  LightNav CasADi/IPOPT MPC；不调用模型 API。标记用例不能验证房间语义推断质量。
- **机器人**：平面动力学底盘具有有限力执行器和碰撞体，不是 Go2 四足关节/步态模型。
- **时序**：仿真按固定时间步同步推进，墙钟时间另记；不是实时性能或网络延迟验收。
- **ROS**：这次绕过 ROS 驱动层，用同一 Observation 协议连接；ROS 节点仍需在 Linux 验证。

报告中的真实目标距离只供评价，绝不传给规划器。视频俯视画面也仅供查看。
下一阶段依次接入真实物体检测、SLAM/ROS 2、Go2 四足模型及其行走控制策略。

macOS 的 MuJoCo 可能提示 `ARB_clip_control unavailable`，表示使用备用深度
渲染路径。本机渲染测试额外检查一个已知平面像素的米制深度误差小于 2 cm；
这不是对所有距离和图形驱动的精度保证。若出现 `invalid CoreGraphics connection`
或 localhost `Operation not permitted`，需要允许测试进程访问图形服务和本机端口。
