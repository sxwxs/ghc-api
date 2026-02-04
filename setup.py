from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ghc-api",
    version="1.0.0",
    author="Medivh",
    author_email="",
    description="GitHub Copilot API Proxy - A Flask application serving as a proxy server for GitHub Copilot API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ghc-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Flask>=2.0.0",
        "requests>=2.25.0",
        "PyYAML>=5.4.0",
        "tiktoken>=0.3.0"
    ],
    entry_points={
        "console_scripts": [
            "ghc-api=ghc_api.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ghc_api": ["templates/*"],
    },
)