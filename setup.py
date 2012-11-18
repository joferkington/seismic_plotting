from setuptools import setup

setup(
    name = 'seismic_plotting',
    version = '0.1',
    description = "Cross Section Plotting with Geoprobe Datasets",
    author = 'Joe Kington',
    author_email = 'joferkington@gmail.com',
    license = 'LICENSE',
    install_requires = [
        'matplotlib >= 0.98',
        'numpy >= 1.1',
        'shapely >= 1.2',
        'scipy >= 0.7',
        ]
)
