import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("ERRORE CRITICO: OPENAI_API_KEY non trovata nel file .env.")
    exit()

client_openai = OpenAI(api_key=openai_key)

def estrai_info_con_openai(testo):
    prompt = f"""Estrai le seguenti informazioni dal testo:
- nome completo
- email
- numero di telefono

Restituisci solo un dizionario JSON nel formato:
{{"nome": "...", "email": "...", "phone": "..."}}

Testo: {testo}
"""
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return "{}"

documents_dir = "resumes"
documents = []
metadatas = []
ids = []

id_counter = 0

print("Caricamento e indicizzazione dei CV...")

if not os.path.exists(documents_dir):
    print(f"Attenzione: La cartella '{documents_dir}' non esiste. La creo adesso in automatico!")
    os.makedirs(documents_dir)
    print("Cartella creata. Inserisci i tuoi file .txt al suo interno e poi riavvia lo script.")
    exit()

for filename in os.listdir(documents_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(documents_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            if content.strip():
                documents.append(content)
                
                info_raw = estrai_info_con_openai(content)
                try:
                    info_json = json.loads(info_raw)
                    info_str = json.dumps(info_json)
                except:
                    info_str = "Info non estratte"
                
                metadatas.append({"source": filename, "info": info_str})
                ids.append(f"doc_{id_counter}")
                id_counter += 1

if not documents:
    print("Nessun documento trovato nella cartella resumes. Aggiungi qualche CV in formato .txt e riprova.")
    exit()

print("Configurazione database vectoriale (ChromaDB)...")

try:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="CVs",
        embedding_function=openai_ef
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Indicizzazione completata con successo.")
except Exception as e:
    print(f"Errore durante la creazione del database vectoriale: {e}")
    exit()

print("\n" + "="*50)
print("SISTEMA RAG PRONTO")
print("="*50)
print("Posso aiutarti a trovare il candidato ideale per qualsiasi ruolo.")
print("Scrivi 'exit' per uscire.\n")

while True:
    user_input = input("Domanda: ")
    if user_input.lower() == 'exit':
        break
    if not user_input.strip():
        continue

    try:
        results = collection.query(
            query_texts=[user_input],
            n_results=1
        )

        if not results['documents'][0] or not results['documents'][0][0]:
            print("Mi spiace, non ho trovato candidati pertinenti per questa richiesta.")
            continue

        doc_content = results['documents'][0][0]
        metadata_info = results['metadatas'][0][0]['info']
        source_file = results['metadatas'][0][0]['source']

        context = f"CONTESTO: nome file {source_file}. Contenuto pertinente: {doc_content}. Dati contatto: {metadata_info}"

        prompt = f"""Dato il seguente contesto:
{context}

Rispondi alla domanda dell'utente: "{user_input}"
Spiega che nel database è presente il profilo più adatto.
Argomenta la scelta utilizzando le competenze descritte nel contesto.
Menziona il nome del candidato e fornisce i dati di contatto alla fine.
"""

        completion = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente HR professionale e preciso. Fornisci risposte utili basandoti solo sui dati forniti."},
                {"role": "user", "content": prompt}
            ]
        )

        print("\n" + "-"*50)
        print(completion.choices[0].message.content)
        print("-"*50 + "\n")

    except Exception as e:
        print(f"Errore durante l'elaborazione della richiesta: {e}")