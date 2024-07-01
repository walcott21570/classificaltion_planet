"""Setup File"""
import os
from setuptools import find_packages, setup


# pylint: disable=exec-used
def get_version():
    """Get package version"""
    version_file = os.path.join("src", "__init__.py")
    with open(version_file, encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("__version__"):
            exec(line.strip())
    return locals()["__version__"]


def get_readme():
    """Parse readme"""
    with open("README.md", encoding="utf-8") as file:
        content = file.read()
    return content


setup(
    name="Modeling",
    version=get_version(),
    author="ASeGordeev",
    description="Modeling HW",
    long_description=get_readme(),
    python_requires=">=3.8",
    packages=find_packages(exclude=(".github", "docs", "examples", "tests")),
)
