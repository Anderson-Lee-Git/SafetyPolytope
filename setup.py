import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="safety_polytope",
    version="0.1.0",
    description="Learning Safety Constraints for Large Language Models (Safety Polytope)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "accelerate==0.30.0",
        "datasets==2.20.0",
        "hydra-core==1.3.2",
        "nltk==3.9.1",
        "omegaconf==2.3.0",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "scipy==1.15.2",
        "seaborn==0.13.2",
        "setuptools==68.2.2",
        "tensorflow==2.18.0",
        "tqdm==4.66.4",
        "transformers==4.45.2",
        "tueplots==0.0.17",
        "umap-learn==0.5.6",
        "vllm==0.6.3.post1",
        "hydra-submitit-launcher==1.2.0",
    ],
    extras_require={
        "dev": [
            "black>=23.1.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
