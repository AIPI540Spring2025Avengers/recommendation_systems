from setuptools import setup, find_packages

setup(
    name="hotel_recommendation_system",
    version="0.1.0",
    description="A deep learning based hotel recommendation system",
    author="AI Team",
    packages=find_packages(include=["scripts", "scripts.*"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "evaluate=Evaluate:evaluate_recommendation_system",
            "train=Train:train_recommendation_system",
            "recommend=Get_Recommandation:get_recommendations",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pth"],
    },
    data_files=[
        ("models", ["models/hotel_tower.pth", "models/user_tower.pth"]),
        ("data", ["data/train.csv"]),
    ],
)
