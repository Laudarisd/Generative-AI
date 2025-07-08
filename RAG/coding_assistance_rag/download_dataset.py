import os
from dotenv import load_dotenv
import requests
from langchain.document_loaders import WebBaseLoader
from kaggle.api.kaggle_api_extended import KaggleApi
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from .env
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Optional for higher rate limits

# Create directory for datasets
os.makedirs("datasets", exist_ok=True)

# 1. Download Python Documentation (Web Scraping)
def download_python_docs():
    url = "https://docs.python.org/3/library/index.html"
    loader = WebBaseLoader(url)
    docs = loader.load()
    with open("datasets/python_docs.txt", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content + "\n")
    print("Downloaded Python documentation to datasets/python_docs.txt")

# 2. Download Stack Overflow Dataset (Kaggle)
def download_stackoverflow_dataset():
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    api = KaggleApi()
    api.authenticate()
    dataset = "stackoverflow/pythonquestions"
    api.dataset_download_files(dataset, path="datasets/stackoverflow", unzip=True)
    print("Downloaded Stack Overflow dataset to datasets/stackoverflow")

# 3. Download GitHub Code Snippets (GitHub API)
def download_github_snippets():
    query = "language:python"
    url = f"https://api.github.com/search/code?q={query}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        with open("datasets/github_snippets.txt", "w", encoding="utf-8") as f:
            for item in data.get("items", []):
                f.write(item["html_url"] + "\n")
        print("Downloaded GitHub snippet URLs to datasets/github_snippets.txt")
    else:
        print(f"Failed to download GitHub snippets: {response.status_code}")

# 4. Download Tutorial Data (DuckDuckGo Search)
def download_tutorial_data():
    with DDGS() as ddgs:
        results = ddgs.text("Python tutorial site:realpython.com", max_results=5)
        urls = [result["href"] for result in results]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    with open("datasets/tutorials.txt", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content + "\n")
    print("Downloaded Python tutorials to datasets/tutorials.txt")

# Run downloads
if __name__ == "__main__":
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env")
    else:
        download_python_docs()
        download_stackoverflow_dataset()
        download_github_snippets()
        download_tutorial_data()
