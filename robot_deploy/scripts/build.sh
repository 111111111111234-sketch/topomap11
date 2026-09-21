#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${script_dir}/.." && pwd)"
lightnav_dir="${1:-}"
if [[ -z "$lightnav_dir" ]]; then
    echo "Usage: bash robot_deploy/scripts/build.sh /absolute/path/to/LightNav-0" >&2
    exit 2
fi
lightnav_dir="$(cd -- "$lightnav_dir" && pwd)"
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "Requires Ubuntu 22.04 and ROS 2 Humble. This does not build on macOS." >&2
    exit 1
fi
for package in vln_mpc robot_adapters/go2_adapter; do
    if [[ ! -f "$lightnav_dir/robot_deploy/src/$package/package.xml" ]]; then
        echo "Missing LightNav package: $package" >&2
        exit 1
    fi
done
source /opt/ros/humble/setup.bash
cd "$workspace_dir"
colcon build --symlink-install --base-paths "$workspace_dir/src" \
    "$lightnav_dir/robot_deploy/src/vln_mpc" \
    "$lightnav_dir/robot_deploy/src/robot_adapters/go2_adapter" \
    --packages-select topomap_go2 vln_mpc go2_adapter
