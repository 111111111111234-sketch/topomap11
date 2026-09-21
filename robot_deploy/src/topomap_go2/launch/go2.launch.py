"""Go2 deployment. Sensor drivers and SLAM are launched/calibrated separately."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(Path(get_package_share_directory("topomap_go2")) / "config" / "go2.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=config),
        DeclareLaunchArgument("allow_motion", default_value="false"),
        DeclareLaunchArgument("connect_robot", default_value="false"),
        DeclareLaunchArgument("network_interface", default_value="eth0"),
        Node(package="topomap_go2", executable="topomap_bridge", output="screen",
             parameters=[LaunchConfiguration("params_file"),
                         {"allow_motion": ParameterValue(LaunchConfiguration("allow_motion"), value_type=bool)}]),
        Node(package="vln_mpc", executable="vln_mpc", output="screen",
             parameters=[{"enabled": False, "objnav_v_max": .2, "track_v_max": .2,
                          "w_max": .4, "a_max_v": .3, "a_max_w": .6,
                          "v_output_scale": 1., "w_output_scale": 1.}],
             remappings=[("mpc/cmd_vel", "topomap/raw_cmd_vel")]),
        Node(package="go2_adapter", executable="go2_adapter", output="screen",
             condition=IfCondition(LaunchConfiguration("connect_robot")),
             parameters=[{"network_interface": LaunchConfiguration("network_interface"), "publish_tf": True}],
             remappings=[("mpc/cmd_vel", "topomap/safe_cmd_vel"),
                         ("web/cmd_vel", "topomap/manual_cmd_vel")]),
    ])
