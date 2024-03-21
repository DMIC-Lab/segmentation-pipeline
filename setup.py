from setuptools import setup, find_packages

setup(
    name='segpipe',
    version='1.0',
    packages=find_packages(),
    description='Description of your project',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        '': ['weights/*'],
    },
    install_requires=[
        'torch',
        'monai[einops]',
        'numpy',
        'scikit-image'
    ],
)
