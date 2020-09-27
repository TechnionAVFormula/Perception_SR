from setuptools import setup, find_packages

setup(
    name="PerceptionSR",
    version="0.1",
    author='Technion Formula Team',
    author_email='technionfs@gmail.com',
    url="https://github.com/TechnionAVFormula/Perception_SR",
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'PerceptionAlgo',
    ],
    entry_points = {
        'console_scripts': [
            'formula-perception-module=PerceptionSR.run_module:main'
        ],
    }

)
