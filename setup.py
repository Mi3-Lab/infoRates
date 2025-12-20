from setuptools import setup, find_packages

setup(
    name="info_rates",
    version="0.1.0",
    description="Temporal sampling analysis for human action recognition",
    author="Wesley Maia",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers>=4.44.0",
        "decord",
        "av",
        "opencv-python",
        "tqdm",
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
    ],
)
