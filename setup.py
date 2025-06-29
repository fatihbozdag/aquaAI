
from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='AquaAI',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A multi-model water quality prediction system with GUI and CLI interfaces.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/AquaAI',  # Replace with your project's URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',  # Choose an appropriate license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    entry_points={
        'gui_scripts': [
            'aqua-ai-gui = main:main',
        ],
        'console_scripts': [
            'aqua-ai-cli = main_cli:main',
        ],
    },
)
