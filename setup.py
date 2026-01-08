import os
import platform
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

profile = os.environ.get("BUILD_PROFILE", "debug").lower()

extra_compile_args = []
extra_link_args = []

if profile == "release":
    extra_compile_args = ["-O3", "-DNDEBUG", "-flto", "-ffast-math", "-ffp-contract=fast"]
    extra_link_args = ["-flto"]
elif profile == "native":
    extra_compile_args = ["-O3", "-DNDEBUG", "-ffast-math", "-ffp-contract=fast"]
    extra_link_args = []
    if sys.platform == "darwin":
        arch = platform.machine()
        if arch:
            extra_compile_args += ["-arch", arch]
            extra_link_args += ["-arch", arch]
else:
    extra_compile_args = ["-O3", "-ffast-math", "-ffp-contract=fast", "-DNDEBUG"]

ext_modules = [
    Pybind11Extension(
        "mcts_ext",
        ["players/mcts_ext.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="mcts_ext",
    version="0.0.0",
    description="Pybind11 scaffold for MCTS engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
