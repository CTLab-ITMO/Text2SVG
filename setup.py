from setuptools import setup, find_packages

def get_requirements():
    with open("requirements.txt") as f:
        # Read lines, strip whitespace, and filter out empty lines/comments
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith(("#", "-"))
        ]

setup(
    name="svgllm",  # Name of your project
    version="0.1.0",
    description="",
    author="Boris Malashenko",
    author_email="quelquemath@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'optimize_dir = optimization.main:main',
        ],
    },
    install_requires=get_requirements()
)