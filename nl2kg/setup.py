from setuptools import setup, find_packages
import os
from glob import glob

package_name = "nl2kg"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "grammars"), glob("grammars/*.gbnf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Miguel Ángel González Santamarta",
    maintainer_email="mgons@unileon.es",
    description="NL2KG: Bidirectional Natural Language Interface for Robot Knowledge Graphs",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nl2kg_node = nl2kg.nl2kg_node:main",
            "nl2kg_cli = nl2kg.nl2kg_cli:main",
            "nl2kg_hri_node = nl2kg.nl2kg_hri_node:main",
        ],
    },
)
