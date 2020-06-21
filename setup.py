from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hmm-lib-SJ",
    version="1.0.0",
    packages=find_packages(exclude=['tests*']),
    license="MIT",
    description="HMM python library",
    long_description=long_description,
    long_description_content_type="test/markdown",
    classifiers=["Programming Language:: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operation System :: OS Independent",
                 ],
    python_requires='>=3.6',
    url="https://github.com/SungjaeJung1031/hmm-lib",
    author="Sungjae Jung",
    author_email="sungjae.jung1031@gmail.com",
)
