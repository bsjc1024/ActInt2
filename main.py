import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    try:
        df = pd.read_csv('SongLyrics.csv') 
    except FileNotFoundError:
        print("Error: Archivo 'SongLyrics.csv' no encontrado.")
        return

    df = df.dropna(subset=['Lyrics', 'Title', 'Artist'])
    
    documents = []
    for _, row in df.iterrows():
        text_content = f"Title: {row['Title']}\nArtist: {row['Artist']}\nLyrics: {row['Lyrics']}"
        metadata = {'title': row['Title'], 'artist': row['Artist']}
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    embeddings = OllamaEmbeddings(model="gemma")

    vector_store = FAISS.from_documents(documents, embeddings)

    query = input("Ingresa un sentimiento, tema o situación (ej. 'nostalgia'): ")
    
    results = vector_store.similarity_search(query, k=5)

    print(f"\nTop 5 canciones para: '{query}'\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.metadata['title']} - {res.metadata['artist']}")

if __name__ == "__main__":
    main()  