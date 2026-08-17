import os

import chromadb
import ollama


class OllamaEmbeddingFunction:

    def name(self):
        return "ollama"

    def __call__(self, input):
        response = ollama.embed(
            model="nomic-embed-text",
            input=input
        )

        return response["embeddings"]

    def embed_query(self, input):
        response = ollama.embed(
            model="nomic-embed-text",
            input=input
        )

        return response["embeddings"]


documents_dir = "resumes"

documents = []
metadatas = []
ids = []

id = 0

for filename in os.listdir(documents_dir):

    if filename.endswith(".txt"):

        file_path = os.path.join(documents_dir, filename)

        with open(file_path, "r", encoding="utf-8") as file:

            text = file.read()

            documents.append(text)

            metadatas.append({
                "source": filename
            })

            ids.append(str(id))

            id += 1


chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="CVs",
    embedding_function=OllamaEmbeddingFunction()
)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)


user_question = "Mi serve qualcuno per promuovere il mio prodotto"

results = collection.query(
    query_texts=[user_question],
    n_results=3
)


print("DOCUMENTI RECUPERATI:\n")

for i in range(3):
    print(f"--- RISULTATO {i + 1} ---")
    print(f"File: {results['metadatas'][0][i]['source']}")
    print(f"Distanza: {results['distances'][0][i]}")
    print(results["documents"][0][i])
    print()


context = ""

for i in range(3):
    context += f"""
Nome file: {results["metadatas"][0][i]["source"]}

Contenuto:
{results["documents"][0][i]}

"""


prompt = f"""
Dato il seguente contesto:

{context}

Rispondi alla domanda dell'utente:

{user_question}

Utilizza esclusivamente le informazioni presenti nel contesto.

Se il contesto non contiene informazioni sufficienti per rispondere,
dillo chiaramente.
"""


print("STO ELABORANDO...")

response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "system",
            "content": (
                "Sei un assistente HR specializzato "
                "nella ricerca di profili professionali."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("\nRISPOSTA RICEVUTA!")
print(response["message"]["content"])