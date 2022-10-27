import setuptools

base_requires = ["jax>=0.3.23", "e3nn-jax>=0.10.1"]
test_requires = ["pytest", "black"]

setuptools.setup(
    name="macx",
    version="0.1.0",
    author="The macx gang",
    author_email="",
    license="ASL",
    install_requires=base_requires,
    extras_require={"test": test_requires},
    packages=setuptools.find_packages(),
    url="https://github.com/sirmarcel/macx",
    python_requires=">=3.8",
)
