from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    description="A local medical chatbot using ChromaDB, HuggingFace embeddings, and Ollama LLM.",
    author="Matheshwara",
    author_email="msenthi7@umd.edu",
    packages=find_packages(where="src"),   # scan only inside src/
    package_dir={"": "src"},               # tell setuptools that code is inside src/
    install_requires=[
        "flask",
        "chromadb",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "pypdf",
        "tiktoken",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
