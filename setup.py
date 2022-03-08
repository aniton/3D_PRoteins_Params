import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


with open("requirements.txt", encoding="utf-8") as file:
    requirements = file.read().splitlines()

name = "3d_solver"


class WithExternal(build_ext):
    def run(self):
        os.system(
            f"pip3 install github-clone"
        )  # for cloning a specific directory of repo
        build_ext.run(self)


setup(
    name=name,
    version="0.0.1",
    packages=find_packages(include=(name,)),
    cmdclass={"build_ext": WithExternal},
    descriprion="Repository of the master thesis part at Skoltech",
    install_requires=requirements,
)
