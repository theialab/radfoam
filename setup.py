import os
import re
import sys
from pathlib import Path

import cmake_build_extension
import setuptools
import sysconfig
import subprocess

lib_path = sysconfig.get_path("purelib")
assert os.path.exists(
    f"{lib_path}/torch"
), "Could not find PyTorch; please make sure it is installed in your environment before installing radfoam."

cmake_options = []

if "CUDA_HOME" in os.environ:
    cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={os.environ['CUDA_HOME']}")

source_dir = Path(__file__).parent.absolute()
cmake = (source_dir / "CMakeLists.txt").read_text()
version = re.search(r"project\(\S+ VERSION (\S+)\)", cmake).group(1)

install_requirements = [
    "cmake==3.29.2",
    "cmake-format",
    "cmake_build_extension",
    "ConfigArgParse",
    "einops",
    "glfw==2.6.5",
    "kornia",
    "opencv-python",
    "pillow",
    "plyfile",
    "pybind11[global]",
    "pyyaml",
    "scipy",
    "tensorboard",
    "tqdm",
]


setuptools.setup(
    version=version,
    install_requires=install_requirements,
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="RadFoamBindings",
            install_prefix="radfoam",
            cmake_depends_on=["pybind11"],
            write_top_level_init=None,
            source_dir=str(source_dir),
            cmake_configure_options=[
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                "-DCALL_FROM_SETUP_PY:BOOL=ON",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DGPU_DEBUG:BOOL=OFF",
                "-DEXAMPLE_WITH_PYBIND11:BOOL=ON",
                f"-DTorch_DIR={lib_path}/torch/share/cmake/Torch",
                "-DPIP_GLFW:BOOL=ON",
            ]
            + cmake_options,
        ),
    ],
    cmdclass=dict(
        build_ext=cmake_build_extension.BuildExtension,
    ),
)
