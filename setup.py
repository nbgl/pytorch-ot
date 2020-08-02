import setuptools


install_requires = [
    'torch>=1.6',
]

setuptools.setup(
    name="torchot",
    version="0.0.1",
    author="Jakub Nabaglo",
    author_email="j@nab.gl",
    description="Differentiable Optimal Transport in PyTorch",
    url="https://github.com/nbgl/pytorch-ot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
)
