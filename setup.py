from setuptools import setup, find_packages

setup(
    name="glabcmcmc",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.1',
        'tqdm>=4.64.0',
        'numpy<2',
        'normflows>=1.7.2',
        'scipy>=1.8.1',
        'pandas>=2.2.2',
        'matplotlib>=3.8.2',
        'seaborn>=0.13.2'
    ],
    author="Xuefei Cao, Shijia Wang, and Yongdao Zhou",
    author_email="xuefeic@mail.nankai.edu.cn, Wangshj1@shanghaitech.edu.cn, and ydzhou@nankai.edu.cn",
    description="A simple Python package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/caofff/GL-ABC-MCMC",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
