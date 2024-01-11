from ragatouille import RAGPretrainedModel
from ragatouille.data import CorpusProcessor, llama_index_sentence_splitter
import os

if __name__ == "__main__":
    RAG = RAGPretrainedModel.from_pretrained(
        "antoinelouis/colbertv1-camembert-base-mmarcoFR"
    )

    path = "/home/adrien/Documents/Coding/RAG_ASN/Ingestion/ASN/lettres_de_suivi/txt"
    full_documents = []
    nb_docs = 200
    for file, i in zip(os.listdir(path), range(0, nb_docs)):
        with open(f"{path}/{file}") as f:
            raw_text = f.read()
        full_documents.append(raw_text)

    RAG.index(
        collection=full_documents,
        index_name="ASN",
        max_document_length=180,
        split_documents=True,
    )
