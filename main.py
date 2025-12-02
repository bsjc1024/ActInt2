import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Para usar Gemma Embeddings localmente
try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print("Instalando dependencias necesarias...")
    os.system("pip install langchain-community ollama")
    from langchain_community.embeddings import OllamaEmbeddings

class SongRecommendationSystem:
    def __init__(self, model_name="gemma:2b"):
        """
        Inicializa el sistema de recomendación con Gemma Embeddings
        
        Args:
            model_name: Nombre del modelo de Ollama (por defecto gemma:2b)
        """
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.songs_df = None
        self.embedding_matrix = None
        self.index_path = "song_index.pkl"
        self.embeddings_path = "song_embeddings.npy"
    
    def cargar_dataset(self, ruta_csv):
        """
        Carga el dataset de Kaggle
        
        Args:
            ruta_csv: Ruta del archivo CSV del dataset
        
        Returns:
            DataFrame con las canciones
        """
        print("Cargando dataset...")
        self.songs_df = pd.read_csv(ruta_csv)
        
        # Asegurarse de que tiene las columnas necesarias
        columnas_requeridas = ['title', 'artist', 'lyrics']
        columnas_presentes = [col for col in columnas_requeridas 
                            if col in self.songs_df.columns]
        
        if len(columnas_presentes) < 3:
            # Intentar mapear nombres alternativos
            print("Ajustando nombres de columnas...")
            if 'song' in self.songs_df.columns:
                self.songs_df.rename(columns={'song': 'title'}, inplace=True)
            if 'artist_name' in self.songs_df.columns:
                self.songs_df.rename(columns={'artist_name': 'artist'}, inplace=True)
            if 'lyric' in self.songs_df.columns:
                self.songs_df.rename(columns={'lyric': 'lyrics'}, inplace=True)
        
        # Eliminar filas con valores nulos en columnas críticas
        self.songs_df = self.songs_df.dropna(subset=['title', 'artist', 'lyrics'])
        
        print(f"Dataset cargado: {len(self.songs_df)} canciones")
        print(f"Columnas disponibles: {list(self.songs_df.columns)}")
        
        return self.songs_df
    
    def generar_embeddings(self, muestra=None, guardar=True):
        """
        Genera embeddings para las letras de las canciones
        
        Args:
            muestra: Número de canciones a procesar (None = todas)
            guardar: Si True, guarda los embeddings en archivo
        """
        print("Generando embeddings con Gemma...")
        
        df_proceso = self.songs_df.head(muestra) if muestra else self.songs_df
        
        embeddings_list = []
        total = len(df_proceso)
        
        for idx, (index, row) in enumerate(df_proceso.iterrows()):
            try:
                # Preparar texto: título + artista + primeros 1000 caracteres de letra
                texto_entrada = f"{row['title']} by {row['artist']}. {row['lyrics'][:1000]}"
                
                # Generar embedding
                embedding = self.embeddings.embed_query(texto_entrada)
                embeddings_list.append(embedding)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Procesadas {idx + 1}/{total} canciones")
                
            except Exception as e:
                print(f"  Error procesando canción {idx}: {str(e)}")
                # Usar embedding de ceros como fallback
                embeddings_list.append([0.0] * 384)  # Gemma produce vectores de 384 dims
        
        self.embedding_matrix = np.array(embeddings_list)
        
        print(f"Embeddings generados: forma {self.embedding_matrix.shape}")
        
        if guardar:
            np.save(self.embeddings_path, self.embedding_matrix)
            self.songs_df.head(muestra).to_pickle(self.index_path)
            print(f"Índice guardado en {self.index_path}")
            print(f"Embeddings guardados en {self.embeddings_path}")
        
        return self.embedding_matrix
    
    def cargar_indice_existente(self):
        """
        Carga índice y embeddings previamente guardados
        """
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            print("Cargando índice existente...")
            self.songs_df = pd.read_pickle(self.index_path)
            self.embedding_matrix = np.load(self.embeddings_path)
            print(f"Índice cargado: {len(self.songs_df)} canciones")
            return True
        return False
    
    def buscar_canciones(self, consulta, top_k=5):
        """
        Busca las canciones más similares a una consulta
        
        Args:
            consulta: Texto de búsqueda (palabra, frase o descripción)
            top_k: Número de canciones a retornar
        
        Returns:
            Lista de tuplas (título, artista, similitud)
        """
        if self.embedding_matrix is None:
            print("Error: No hay embeddings cargados. Genera o carga un índice primero.")
            return []
        
        print(f"\nBuscando canciones para: '{consulta}'")
        
        # Generar embedding para la consulta
        try:
            embedding_consulta = self.embeddings.embed_query(consulta)
            embedding_consulta = np.array(embedding_consulta).reshape(1, -1)
        except Exception as e:
            print(f"Error generando embedding de consulta: {e}")
            return []
        
        # Calcular similitud coseno
        similitudes = cosine_similarity(embedding_consulta, self.embedding_matrix)[0]
        
        # Obtener índices de top-k
        indices_top = np.argsort(similitudes)[::-1][:top_k]
        
        resultados = []
        for i, idx in enumerate(indices_top):
            cancion = self.songs_df.iloc[idx]
            similitud = similitudes[idx]
            resultados.append({
                'ranking': i + 1,
                'titulo': cancion['title'],
                'artista': cancion['artist'],
                'similitud': round(similitud, 4)
            })
        
        return resultados
    
    def mostrar_resultados(self, resultados):
        """
        Muestra los resultados de forma formateada
        """
        if not resultados:
            print("No se encontraron resultados.")
            return
        
        print("\n" + "="*70)
        print("TOP 5 CANCIONES MÁS RELEVANTES")
        print("="*70)
        
        for r in resultados:
            print(f"\n{r['ranking']}. {r['titulo']}")
            print(f"   Artista: {r['artista']}")
            print(f"   Similitud: {r['similitud']:.1%}")
        
        print("\n" + "="*70)


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    
    # 1. Crear instancia del sistema
    sistema = SongRecommendationSystem(model_name="gemma:2b")
    
    # 2. Cargar dataset (descargar desde Kaggle primero)
    # Descargar: https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset
    ruta_dataset = "song_lyrics.csv"  # Cambiar por tu ruta
    
    if os.path.exists(ruta_dataset):
        sistema.cargar_dataset(ruta_dataset)
        
        # 3. Generar embeddings (primera vez)
        # Usar muestra=100 para prueba rápida
        sistema.generar_embeddings(muestra=100)
        
    elif sistema.cargar_indice_existente():
        print("Usando índice existente...")
    else:
        print("Por favor, descarga el dataset de Kaggle y colócalo como 'song_lyrics.csv'")
        print("URL: https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset")
    
    # 4. Realizar búsquedas
    if sistema.songs_df is not None and sistema.embedding_matrix is not None:
        
        consultas = [
            "nostalgia y tristeza",
            "ruptura amorosa y dolor",
            "superación personal",
            "tarde lluviosa melancólica",
            "felicidad y amor"
        ]
        
        for consulta in consultas:
            resultados = sistema.buscar_canciones(consulta, top_k=5)
            sistema.mostrar_resultados(resultados)
            print("\n")