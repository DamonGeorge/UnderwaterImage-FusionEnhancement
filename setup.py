from setuptools import setup

# get current version
exec(open("fusion_enhance/_version.py").read())

setup(
    name="FusionEnhance",
    url="https://github.com/DamonGeorge/UnderwaterImage-FusionEnhancement",
    author="Damon George",
    author_email="damon@kindgeorge.com",
    packages=["fusion_enhance"],
    install_requires=["numpy", "opencv-python"],
    version=__version__,  # pylint: disable=E0602
    license="MIT",
    description="Python implementation of a fusion enhancement algorithm for underwater images.",
    long_description=open("README.md").read(),
)
