import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    print("Leyendo SongLyrics.csv...")
    try:
        df = pd.read_csv("SongLyrics.csv")
    except FileNotFoundError:
        print("Error: Archivo 'SongLyrics.csv' no encontrado en la carpeta del proyecto.")
        return

    print(f"Filas totales en el CSV: {len(df)}")

    df = df.dropna(subset=["Artist", "Title", "Lyrics"])
    print(f"Filas después de dropna: {len(df)}")

    # usar solo una parte del dataset para que no tarde tanto
    df = df.head(10) # comentar si no se utiliza
    print(f"Usando {len(df)} canciones para el índice.")

    documents = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        doc = Document(
            page_content=row["Lyrics"],
            metadata={
                "title": row["Title"],
                "artist": row["Artist"],
            },
        )
        documents.append(doc)
        if i % 50 == 0:
            print(f"  Preparadas {i} canciones...")

    print("Creando objeto de embeddings con Ollama...")
    
    embeddings = OllamaEmbeddings(model="gemma")
    print("Embeddings listos (conexión a Ollama OK).")

    # índice FAISS
    print("Construyendo índice FAISS (esto puede tardar un poco)...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Índice FAISS creado. Ya puedes hacer consultas.\n")

    # consulta
    while True:
        query = input("Ingresa un sentimiento, tema o situación (o 'salir'): ")
        if query.lower().strip() == "salir":
            print("Adiós ✨")
            break

        results = vector_store.similarity_search(query, k=5)

        print(f"\nTop 5 canciones para: '{query}'\n")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res.metadata['title']} - {res.metadata['artist']}")
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()
