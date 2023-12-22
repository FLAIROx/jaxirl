from setuptools import setup, find_packages

setup(
    name="jaxirl",
    version="0.1.0",
    description="Implementation of IRL algorithms in JAX",
    author="Silvia Sapora",
    author_email="silvia.sapora@gmail.com",
    packages=find_packages(exclude=("tests")),
)

print(find_packages(exclude=("tests")))
