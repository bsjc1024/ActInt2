# Actividad Integradora 2 – Buscador Semántico de Canciones

Este proyecto construye un **motor de búsqueda semántica de canciones**.  
A partir de un dataset de letras, se crea un índice vectorial con **FAISS** y **embeddings de Gemma vía Ollama**, para que puedas consultar canciones por sentimiento, tema o situación (por ejemplo: `tristeza`, `motivación`, `nostalgia`).

---

## Integrantes

- Bum Soo Jang – A01665654  
- Pedro Luis Castañeda Pastelín – A01736619  
- Isaac Martínez Trujillo – A01735069  

Profesor: Luciano García Bañuelos  
Materia: Análisis y diseño de algoritmos avanzados  
Fecha de entrega: 04 de diciembre del 2025

---

## 1. Requisitos

- macOS (probado en Mac M1 / Apple Silicon).
- [Homebrew](https://brew.sh/) instalado.
- Python 3.10+ (`python3 --version` para comprobar).
- Conexión a internet para descargar:
  - Ollama  
  - El modelo `gemma`
  - Paquetes de Python
- Espacio en disco (el modelo Gemma ocupa varios GB).

---

## 2. Instalación de Ollama con Homebrew

En una terminal:

```bash
brew update
brew install ollama
```

Comprueba que quedó bien instalado:

```bash
ollama --version
```

Si ves un número de versión, todo está OK.

Luego descarga el modelo **Gemma**:

```bash
ollama pull gemma
# o una variante específica, por ejemplo:
# ollama pull gemma:2b
```

> Mantén Ollama en ejecución (si lo instalaste como servicio se levanta solo; si no, puedes usar `ollama serve` en otra terminal).

---

## 3. Preparar el entorno de Python

1. **Clona o descomprime el proyecto**

   Supongamos que el repositorio/zip se descomprime en:

   ```bash
   ~/Documents/ActInt2
   ```

   Entra a esa carpeta:

   ```bash
   cd ~/Documents/ActInt2
   ```

2. **Crea un entorno virtual**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   (Cada vez que trabajes con el proyecto, activa el entorno con `source .venv/bin/activate`).

3. **Instala las dependencias de Python**

   ```bash
   pip install --upgrade pip
   pip install pandas langchain langchain-core langchain-community langchain-ollama faiss-cpu
   ```

   > Si `faiss-cpu` diera problemas, puedes instalarlo con Conda:
   >
   > ```bash
   > conda create -n actint2 python=3.11
   > conda activate actint2
   > conda install -c conda-forge faiss-cpu
   > pip install pandas langchain langchain-core langchain-community langchain-ollama
   > ```

---

## 4. Estructura del proyecto

Dentro de la carpeta principal deberías tener algo parecido a esto:

```text
ActInt2/
├── main.py
├── Song Lyrics Dataset/
│   └── csv/
│       ├── ArianaGrande.csv
│       ├── BTS.csv
│       ├── ...
└── SongLyrics.csv
```

- `Song Lyrics Dataset/csv/` contiene muchos CSV, uno por artista.
- `main.py` construye el índice semántico y permite hacer consultas.

---

## 5. Ejecutar el buscador semántico

Con todo listo:

1. Asegúrate de estar en el entorno virtual y la carpeta del proyecto:

   ```bash
   cd ~/Documents/ActInt2
   source .venv/bin/activate
   ```

2. Lanza el programa:

   ```bash
   python main.py
   ```

El flujo típico será:

1. Carga de `SongLyrics.csv`
2. Limpieza de datos (`dropna` en `Artist`, `Title`, `Lyrics`)
3. (Opcional, en tu versión) seleccionar una muestra del dataset para acelerar pruebas:
   ```python
   df = df.sample(n=200, random_state=42)  # o df.head(200)
   ```
4. Construcción de documentos y generación de embeddings con Gemma.
5. Creación del índice FAISS.
6. Pregunta en consola:

   ```text
   Ingresa un sentimiento, tema o situación (ej. 'nostalgia'):
   ```

Escribe, por ejemplo:

```text
tristeza
```

La salida esperada:

```text
Top 5 canciones para: 'tristeza'

1. Título 1 - Artista 1
2. Título 2 - Artista 2
3. ...
```

---

## 8. Errores comunes y soluciones

### `KeyError: ['Lyrics']` o columnas faltantes

- Ocurre cuando el CSV tiene la columna `Lyric` y no `Lyrics`.
- Re-genera el CSV con `crear_csv.py` actualizado (que renombra `Lyric` → `Lyrics`), o corrige a mano el encabezado del archivo.

### El programa parece “congelado” después de `OllamaEmbeddings(model="gemma")`

- No está colgado: está calculando embeddings para **muchas canciones**.
- Usa solo una parte del dataset mientras desarrollas:

  ```python
  df = df.sample(n=200, random_state=42)  # o df.head(200)
  ```

- Agrega `print()` de progreso para ver que sigue vivo.

### Problemas con `faiss-cpu`

- Si la instalación con `pip` falla en Apple Silicon, usa la alternativa con Conda descrita en la sección de dependencias.

---

## 9. Cómo extender el proyecto

Algunas ideas rápidas:

- Cambiar el modelo de embeddings (otro modelo de Ollama).
- Guardar el índice FAISS en disco y recargarlo para no recalcular embeddings cada vez.
- Crear una pequeña API (por ejemplo con FastAPI) o interfaz web para consultas.
- Mostrar también fragmentos de la letra en los resultados.

---

## 10. Créditos

Proyecto desarrollado como **Actividad Integradora 2**, demostrando:

- Uso de procesamiento de lenguaje natural con modelos locales (Gemma vía Ollama).
- Búsqueda semántica basada en vectores con FAISS.
- Integración de Python, LangChain y un dataset real de letras de canciones.
