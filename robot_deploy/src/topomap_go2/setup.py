from glob import glob
from setuptools import setup

setup(
    name="topomap_go2", version="0.1.0", packages=["topomap_go2"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/topomap_go2"]),
        ("share/topomap_go2", ["package.xml"]),
        ("share/topomap_go2/launch", glob("launch/*.launch.py")),
        ("share/topomap_go2/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    entry_points={"console_scripts": ["topomap_bridge = topomap_go2.node:main"]},
)
