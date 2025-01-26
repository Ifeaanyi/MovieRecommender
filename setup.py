from setuptools import setup, find_packages

# Reading the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define constants
AUTHOR_NAME = 'Adika Ifeanyi'
SRC_REPO = 'src'
LIST_OF_REQUIREMENTS = ['streamlit']

# Setup function
setup(
    name=SRC_REPO,  # Package name
    version='0.0.1',  # Version of the package
    author=AUTHOR_NAME,  # Author's name
    author_email='adika.emeka@gmail.com',  # Author's email
    description='Creating a simple Python package for a movie recommendation system',  # Short description
    long_description=long_description,  # Long description from README.md
    long_description_content_type='text/markdown',  # Description type
    packages=find_packages(where=SRC_REPO),  # Automatically find packages in the specified directory
    python_requires='>=3.7',  # Minimum Python version required
    install_requires=LIST_OF_REQUIREMENTS,  # List of dependencies
    url='https://github.com/Ifeaanyi/Movie-Reccomendation-system',  # Repository URL
)
