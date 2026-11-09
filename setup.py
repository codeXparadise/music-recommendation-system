from setuptools import setup, find_packages

setup(
    name="music-recommendation-system",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit>=1.35",
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
    ],
    entry_points={
        "console_scripts": [
            "tunematch=music_recommender.cli:main",
        ],
    },
)

