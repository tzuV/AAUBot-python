"""Document ingestion: download Python docs, parse HTML, chunk, embed, store."""
import os
import urllib.request
import zipfile

from bs4 import BeautifulSoup


def download_docs(url: str, cache_path: str) -> str:
    """Download and extract the Python docs zip. Returns path to extracted docs."""
    extract_path = os.path.join(cache_path, "extracted")
    if os.path.isdir(extract_path) and any(os.scandir(extract_path)):
        print("  Docs already extracted, skipping download.")
        return extract_path

    os.makedirs(cache_path, exist_ok=True)
    zip_path = os.path.join(cache_path, "docs.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading docs from {url}...")
        urllib.request.urlretrieve(url, zip_path)

    print("  Extracting docs...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    return extract_path


def parse_html_files(extract_path: str) -> list[dict]:
    """Parse all HTML files. Returns list of {'text', 'title', 'url'}."""
    docs = []
    html_files = []
    for root, _, files in os.walk(extract_path):
        for f in files:
            if f.endswith(".html"):
                html_files.append(os.path.join(root, f))

    print(f"  Found {len(html_files)} HTML files")

    for filepath in html_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""

        main = soup.find("main") or soup.find("div", class_="body") or soup.body
        if not main:
            continue
        text = main.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) < 50:
            continue

        # Construct source URL from relative path
        rel_path = os.path.relpath(filepath, extract_path)
        parts = rel_path.replace(os.sep, "/").split("/")
        if parts and parts[0].startswith("python-"):
            rel_url = "/".join(parts[1:])
        else:
            rel_url = rel_path.replace(os.sep, "/")
        url = f"https://docs.python.org/3/{rel_url}"

        docs.append({"text": text, "title": title, "url": url})

    return docs


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into chunks, breaking on paragraph boundaries where possible."""
    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current += para + "\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                current = ""
            else:
                current = para + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def ingest(docs_url: str, cache_path: str, embedder, retriever,
           chunk_size: int, chunk_overlap: int) -> int:
    """Full ingest pipeline. Returns number of chunks stored."""
    if retriever.count() > 0:
        print(f"  ChromaDB already has {retriever.count()} chunks. Skipping ingest.")
        return retriever.count()

    extract_path = download_docs(docs_url, cache_path)
    docs = parse_html_files(extract_path)
    print(f"  Parsed {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "title": doc["title"],
                "url": doc["url"],
                "id": f"{doc['url']}_{j}",
            })

    print(f"  Created {len(all_chunks)} chunks")

    batch_size = 256
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed(texts)
        ids = [c["id"] for c in batch]
        metadatas = [{"title": c["title"], "url": c["url"]} for c in batch]
        retriever.add(ids, embeddings, texts, metadatas)
        batch_num = i // batch_size + 1
        print(f"  Ingested batch {batch_num}/{total_batches}")

    print(f"  Ingest complete: {retriever.count()} chunks in ChromaDB")
    return retriever.count()
