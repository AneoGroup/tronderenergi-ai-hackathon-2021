from setuptools import find_packages, setup

version = "0.0.1"

setup(
    name="tekml-rye-flex-gym",
    version=version,
    description="Environment for power dynamics at Rye microgrid ",
    long_description_content_type="text/markdown",
    author="TrønderEnergi AI Team",
    author_email="",
    url="https://github.com/TronderEnergi/hackathon-brain-2021-rye-microgrid-flex",
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,  # https://mypy.readthedocs.io/en/latest/installed_packages.html
    packages=find_packages("src"),
    install_requires=["gym>=0.18.0", "matplotlib>=1.1.0", "pandas>=1.2.3"],
    extras_require={"dev": ["black", "pytest", "isort", "tox"]},
    python_requires=">=3.7",
)
