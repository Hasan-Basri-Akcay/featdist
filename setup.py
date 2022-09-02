import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featdist",
    version="v0.1.2",
    author="Hasan Basri Akcay",
    author_email="hasan.basri.akcay@gmail.com",
    description="Train test target distribution function for machine learning",
    long_description=(
        "Featdist (Train Test Target Distribution) helps with feature understanding, "
        "calculating feature importances, feature comparison, feature debugging, and "
        "leakage detection"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/Hasan-Basri-Akcay/featdist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy", "matplotlib"],
    keywords=["python", "data science", "data analysis", "exploratory data analysis", 
              "distribution", "beginner"],
)
