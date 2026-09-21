# topomap · Go2 首版部署

本目录实现一个 **独立的 Go2 类别目标导航后端**：ROS RGB-D/定位/地图输入 →
YOLO-World 目标检测与深度定位 → HGR 假设图与前沿选择 → 已知自由空间 A* →
LightNav MPC → Go2。支持目标类别，例如 `chair`、`sofa`、`refrigerator`。

**当前是代码接入版，尚未在真实 Go2、ROS 2 或真实模型上完成闭环验收。**
默认只观察并输出规划结果，不连接机器人、不允许运动。它不是把整个
`run_goatbench_evaluation.py` 搬到机器人上：本版使用外部 SLAM 的占据地图，
复用现有 `HypothesisGraph`、`HypothesisNodePredictor`、`SemanticCritic`、
`route_guidance.grid_path`；不包含原实验的 ConceptGraph/TSDF 流程、所有双拓扑
消融版本、图像目标、任意长指令、动态跟踪或多楼层导航。物体位置目前使用检测框
中心区域的深度中位数，不是实例分割结果；同类多个物体不区分实例身份。

Mac 导航闭环可先按 [MuJoCo 仿真说明](../deploy/sim/README.md) 运行：使用实际
RGB-D 渲染、深度建图、HGR/规划、HTTP、MPC 和平面接触动力学。它仍不代替上述
真实模型、ROS 2 或 Go2 四足整机验收。

## 1. 两侧环境

| 位置 | 环境 | 职责 |
| --- | --- | --- |
| GPU 主机 | 建议 Linux、Python 3.10、可用 CUDA | YOLO-World、HGR、HTTP 导航服务 |
| Go2 上位机 | Ubuntu 22.04、ROS 2 Humble、Python 3.10 | RGB-D、SLAM、桥接、MPC、Unitree SDK |
| 当前 Mac | Python 3.11 独立仿真环境 | 配置/CPU 测试、诊断标记导航闭环；非四足步态 |

不要把 Habitat 的 Python 3.9 环境混进 ROS 环境。本后端不安装/导入 Habitat、
Open3D 或 PyTorch3D。GPU 主机与机器人之间推荐使用 SSH 隧道。

### GPU 主机

在 topomap 根目录执行：

```bash
python3.10 -m venv .venv-go2
source .venv-go2/bin/activate
python -m pip install --upgrade 'pip>=23'
pip install -e '.[server]'
```

根据 GPU 安装匹配的 PyTorch/CUDA wheel，然后用
`python -c 'import torch; print(torch.cuda.is_available())'` 检查。
将官方 YOLO-World 的 `yolov8x-world.pt` 放到 `checkpoints/`，或者修改
`deploy/go2/server.json` 的 `yolo_weights`（相对启动目录）。
Ultralytics 的 `set_classes` 还会用到 CLIP 文本编码器及其权重；首次配置类别
可能触发依赖/权重下载，应在联网准备阶段完成。不要将未经准备的首次启动当成
离线可运行安装。

默认 VLM 是 DashScope 的 `qwen3-vl-plus`，设置 `DASHSCOPE_API_KEY` 后启动：

```bash
topomap-go2-serve --config deploy/go2/server.json --check-config
topomap-go2-serve --config deploy/go2/server.json
```

API 凭据通过环境变量传入，不写配置文件。可修改 JSON 中的 VLM 提供商配置。
如暂时没有 API，将 `enable_vlm_hypothesis_prediction` 显式设为 `false`：
这是**启发式语义预测模式**，不要将它的表现记为真实 VLM 模式。
现有 HGR predictor 在 VLM 请求失败时也可能回退到启发式，服务器日志会记录。
视觉特征残差权重设为零，因为本版没有加载单独的 CLIP 图像证据编码器；仍执行
类别和物体残差检查。

服务监听 `127.0.0.1:8060`，加载模型后才开放端口。每进程支持一台机器人，
一个活动任务；`reset` 会清空该任务记忆。不是 LightNav 的 WebSocket 协议。

在 Go2 上位机建立隧道（替换用户名和 GPU 地址）：

```bash
ssh -N -L 8060:127.0.0.1:8060 user@gpu-host
```

桥接节点连接 `http://127.0.0.1:8060`。需要直接绑定远端地址时，服务器和
客户端同时设置 `TOPOMAP_TOKEN`，服务端加 `--host 0.0.0.0`；HTTP 只用于可信
内网，跨网络使用加密隧道。

### Go2 上位机

先安装 ROS 2 Humble、`colcon`、`rosdep`、相机驱动和 RTAB-Map ROS。
Unitree SDK 与 CycloneDDS 安装要求见本地 LightNav 的
`robot_deploy/src/robot_adapters/go2_adapter/README.md`。实际 Go2 型号/固件必须
开放 `rt/lf/sportmodestate`、`rt/lowstate` 和运动 API；不能由“Go2”型号名称
推断所有版本都支持。

设 `LIGHTNAV_DIR` 为 **Linux 上** 的 LightNav checkout 路径：

```bash
export LIGHTNAV_DIR=/absolute/path/to/LightNav-0
source /opt/ros/humble/setup.bash
rosdep install --from-paths robot_deploy/src \
  "$LIGHTNAV_DIR/robot_deploy/src/vln_mpc" \
  "$LIGHTNAV_DIR/robot_deploy/src/robot_adapters/go2_adapter" --ignore-src -r -y
python3 -m pip install --user -e . 'casadi>=3.7,<4'
bash robot_deploy/scripts/build.sh "$LIGHTNAV_DIR"
source robot_deploy/install/setup.bash
```

构建脚本引用现有 LightNav 源码，构建产物放在本项目 `robot_deploy` 下，不复制
或改写 LightNav。相关代码仍遵循 LightNav 的原许可。`pyproject.toml` 的基本
安装不含模型依赖，机器人侧不需要安装 `[server]`。

## 2. 必须准备的传感器与定位

相机型号尚未确定，因此 launch **不会假设内置 RGB 相机有深度，也不会启动未知
型号的相机驱动或填入虚构外参**。先用相机自己的驱动建立下列接口，再修改
`src/topomap_go2/config/go2.yaml` 中的话题名：

| 输入 | 默认话题/约定 |
| --- | --- |
| 去畸变 RGB | `/camera/color/image_rect`，`rgb8` 或可转换格式 |
| 对齐到 RGB 的深度 | `/camera/aligned_depth_to_color/image_raw`，`16UC1` 毫米或 `32FC1` 米 |
| RGB 内参 | `/camera/color/camera_info`，尺寸一致，使用 rectified P 矩阵 |
| 地图 | `/rtabmap/map`，`nav_msgs/OccupancyGrid`，`frame_id=map` |
| Go2 里程计 | `/odom`，`odom → base_link`，与 MPC 使用同一来源 |
| TF | `map → odom → base_link → RGB optical frame` |

建议先用 640×480 RGB-D（协议上限 1280×720），两幅图与 CameraInfo 必须使用
同一个 RGB optical frame。深度必须已经注册到去畸变 RGB，改话题名不能替代对齐。
IMU 不直接上传本服务，由 SLAM/里程计模块消费。

Go2 适配器可发布 `odom → base_link` TF。标定并发布 `base_link → camera_link`，
相机驱动通常发布 `camera_link → camera_*_optical_frame`；不要重复发布同一 TF。
ROS 机身坐标是 x 前/y 左/z 上，光学坐标是 x 右/y 下/z 前。服务使用图像时间戳
查询 TF，不能拿“最新 TF”替代拍摄时刻位姿。
图像先在有界队列等待 `tf_wait_s`（默认 80 ms），让同一时刻的里程计先到达；
这不会阻塞 ROS 回调。仍缺少对应 TF 时会停止，不外推一个虚构位姿。

首版定位建议 **Go2 里程计 + RTAB-Map RGB-D 建图/回环**。下面是常见
`rtabmap_launch` 的配置示例，安装版本、相机话题及消息 QoS 需在现场核对：

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  frame_id:=base_link visual_odometry:=false odom_topic:=/odom \
  rgb_topic:=/camera/color/image_rect \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/color/camera_info approx_sync:=true qos:=2 \
  rtabmap_args:="--Grid/FromDepth true --Grid/CellSize 0.05 --Grid/RangeMax 5.0"
```

须检查地面分割、障碍物高度、尺度、回环和地图发布频率。地图默认 3 秒未更新
即停止；若你的地图只在变化时发布，需要配置周期重发/合理超时并实际测量失联
行为。`map → odom` 单次平移跳变超过 0.3 m 或偏航跳变超过 0.25 rad 时，桥接
节点会撤销任务并解除运动使能；检查回环结果后重新发目标，避免旧记忆错位。

## 3. 先只观察

已由其他进程发布里程计时，不要再启动第二个 Go2 适配器：

```bash
ros2 launch topomap_go2 go2.launch.py params_file:=/absolute/path/to/go2.yaml
```

如需本 launch 连接 Go2 读取状态（**这些命令由操作者现场执行**）：

```bash
ros2 launch topomap_go2 go2.launch.py connect_robot:=true \
  network_interface:=eth0 params_file:=/absolute/path/to/go2.yaml
```

这仍是 `allow_motion=false`。设置目标并观察：

```bash
ros2 topic pub --once /topomap/goal std_msgs/msg/String "{data: chair}"
ros2 topic echo /topomap/status
ros2 topic echo /topomap/shadow_response
```

只有配置 `classes` 内的单个英文类别有效；当前不接受“先去厨房再找红椅子”。
先手动扫描一小片区域，让地图中机器人整个足迹及前方路径成为已知自由空间。
空地图/未知脚下区域不会被自动填成可走，也不会自动原地旋转找路。

验证目标类别与 RGB-D 对齐、路径方向、米制尺度、地图占据和实际物体一致。
`RUNNING` 表示规划有路径，**不代表机器人已经运动**；观察模式的结果只发布到
`topomap/shadow_response`。`BLOCKED` 不是“任务成功”，只有连续三帧当前深度支持
的目标检测且进入距离阈值，才返回 `ARRIVED`。

## 4. 低速真机验收

在完成上一阶段并由操作者准备好 Go2 后，重新启动 launch，加
`allow_motion:=true connect_robot:=true`。默认仍未武装，不会自行切到自动控制。
操作者通过原厂遥控器完成站立/行走准备，发目标，再显式启用：

```bash
ros2 topic pub --once /topomap/goal std_msgs/msg/String "{data: chair}"
ros2 service call /control/set_auto std_srvs/srv/Trigger '{}'
ros2 service call /topomap/enable_motion std_srvs/srv/SetBool '{data: true}'
```

默认速度上限 0.2 m/s、角速度 0.4 rad/s；MPC 输出倍率为 1。
圆形规划足迹半径默认 0.4 m，另加栅格离散余量，需按照实机尺寸、安装件和环境
核对。服务端与机器人 YAML 的 `robot_radius_m` 必须一致。

停止并释放自动控制：

```bash
ros2 service call /topomap/enable_motion std_srvs/srv/SetBool '{data: false}'
ros2 topic pub --once /topomap/goal std_msgs/msg/String "{data: ''}"
ros2 service call /control/stop std_srvs/srv/Trigger '{}'
```

自主命令只有一条链：

```text
vln/response → LightNav MPC → topomap/raw_cmd_vel
                                  ↓
       时间戳/传感器/路径有效期/深度障碍/限速门控
                                  ↓
                 topomap/safe_cmd_vel → Go2 适配器
```

不要同时运行 LightNav 原版 `go2.launch.py` 或把 raw_cmd_vel 直接接机器人。
手动命令另映射到 `topomap/manual_cmd_vel`，本版不自动启动 Web 驾驶面板。
原厂遥控器和适配器的模式仲裁仍适用。禁止把自动控制链的测试当作整机安全认证。

桥接节点在图像/里程计过期、地图过期、规划过期、结果与请求不匹配、地图路径
失效或前方深度障碍时发零速度并禁用 MPC；进程崩溃依靠 Go2 适配器的命令
watchdog。深度前方停机体积为前方 0.75 m、左右 0.3 m、相对机身高度
[-0.15, 0.6] m，只是补充检查，不覆盖相机盲区、侧向障碍、玻璃或落差。
Go2 原厂避障仍须保留，并在现场测试，初版限定平坦室内地面。

VLM 初始化预测可能较慢。旧帧结果不会直接驱动机器人；下一轮可使用已缓存
的语义假设和新观测重规划。结果默认超过 2 秒作废，路径只执行 1.5 秒后需刷新。
先测延迟再调整频率，不能仅为“让机器人动起来”无限延长过期阈值。

必须逐项实测：已知短路径跟踪、目标到达、未知区域拒绝、临时障碍、拔相机、
停定位、断服务网络、任务取消、回环跳变、进程退出和原厂遥控接管。

## 5. 本机离线验证

```bash
python3 -m venv .venv-go2
.venv-go2/bin/python -m pip install -e '.[test]'
.venv-go2/bin/python -m pytest -q tests/test_go2_deploy.py tests/test_go2_bridge.py tests/test_go2_http.py
.venv-go2/bin/python -m topomap_deploy.server --config deploy/go2/server.json --check-config
```

测试使用真实协议、规划、图结构和桥接控制代码，传感器/检测模型/ROS 消息接口
使用离线替身。通过只说明代码契约与保护逻辑满足这些测试，不能说明 ROS 驱动、
GPU 模型或 Go2 已经部署成功。

## 6. 服务协议

- `GET /health`：模型加载后的就绪状态、支持类别。
- `POST /reset`：`{"goal":"chair"}` → `{"session":"...","goal":"chair"}`。
- `POST /step`：`{"session":"...","observation":...}`。
- `POST /stop`：`{"session":"..."}`，清空活动任务。

Observation v1 的编解码在 `topomap_deploy/protocol.py`：JPEG RGB；小端
float32 米制深度；int8 ROS 占据图；相机内参；捕获时刻的 map←base_link、
map←camera_optical 刚体变换；递增序号与纳秒时间戳。最多 16 MiB/请求，最多
100 万地图单元。坐标地图原点包含 yaw，不假设原点是零。

返回 `[forward_m, left_m, yaw_rad]` 是**拍摄时刻机身坐标系中的绝对局部轨迹**，
不能逐项累加成增量。桥接保留 `capture_stamp_ns`，由 LightNav MPC 使用
同源 Go2 里程计在拍摄时刻对齐。换目标、任务取消后的旧响应会被丢弃。
