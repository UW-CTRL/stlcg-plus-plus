from setuptools import setup

setup(
    name="stlcgpp",
    version="0.0.2",
    description="stlcg++ with pytorch",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["stlcgpp"],
    install_requires=[
        "torch>=2.6.0",
        "matplotlib",
        "numpy",
        "graphviz"
    ],
)