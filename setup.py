import glob
from setuptools import find_packages, setup

package_name = "dynamic_tracking"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
    ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
    ("share/" + package_name + "/rviz", glob.glob("rviz/*.rviz")),
    ("share/" + package_name + "/models/spencer_rf", glob.glob("models/spencer_rf/*.gz")),
    ("share/" + package_name + "/models/spencer_adaboost", glob.glob("models/spencer_adaboost/*.*")),
    ("share/" + package_name + "/models/yolo", glob.glob("models/yolo/*.*")),
    ("share/" + package_name + "/scripts", glob.glob("scripts/*.sh")),
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Edison Bejarano — PhD at IRI Mobile robotics group",
    maintainer_email="edison.bejarano@upc.edu",
    description="Dynamic object detection for ROS2 with YOLO and experimental LiDAR fusion",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dynamic_tracking_node = dynamic_tracking.dynamic_tracking_node:main",
        ],
    },
)
