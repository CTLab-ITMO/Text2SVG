from setuptools import setup, find_packages

def get_requirements():
    with open("requirements.txt") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith(("#", "-"))
        ]

setup(
    name="svg-corpus",
    version="0.1.0",
    description="",
    author="Boris Malashenko",
    author_email="btmalashenko@itmo.ru",
    packages=find_packages(where="svg-corpus"),
    package_dir={"": "svg-corpus"},
    entry_points={
        'console_scripts': [
            'optimize_svg_corpus = optimization.main:main',
            'caption_dir = captioning.main:main',
            'mining_dir = mining.main:main'
        ],
    },
    install_requires=get_requirements()
)
