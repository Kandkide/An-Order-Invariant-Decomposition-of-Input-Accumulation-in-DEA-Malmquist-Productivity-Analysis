from setuptools import setup, find_packages

setup(
    name="my_package",  # パッケージ名（自由に設定）
    version="0.1.0",  # バージョン
    packages=find_packages(),  # `my_package`ディレクトリを自動検出
    install_requires=[],  # 必要な依存パッケージをリストで記述
    author="Your Name",  # 作者名
    author_email="your_email@example.com",  # メールアドレス
    description="A collection of Python scripts",  # 簡単な説明
    long_description=open("README.md").read(),  # 詳細な説明をREADMEから取得
    long_description_content_type="text/markdown",  # README形式を指定
    url="https://github.com/your_username/my_package",  # 任意のURL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Pythonのバージョン要件
)
