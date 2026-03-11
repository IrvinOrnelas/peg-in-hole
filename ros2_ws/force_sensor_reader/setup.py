from setuptools import find_packages, setup
import os
from glob import glob

package_name = "force_sensor_reader"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="ROS2 Humble serial reader for force_sensor Arduino sketch",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "force_sensor_node = force_sensor_reader.force_sensor_node:main",
        ],
    },
)
