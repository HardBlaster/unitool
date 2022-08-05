from setuptools import find_packages, setup

setup(
    name='universal_tool',
    packages=find_packages(include=['universe']),
    version='0.1.0',
    description='Collection of methods used frequently in ML related projects.',
    author='Hidi Erik',
    license='MIT',
    install_requires=['pandas', 'numpy', 'matplotlib', 'opencv-python'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
