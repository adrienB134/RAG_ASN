# RAG_ASN
## 🚧 Building an end to end RAG pipeline 🚧
<br>
The goal is to build a full Retrieval Augmented Generation pipeline, composed of the following steps: <br>
* Scraping the data<br>
* Processing it and storing it in a vector store<br>
* Chat app doing the retrieval and displaying it.<br>
<br>

## Scraping

Done with Scrapy, see scraping folder.

## Ingestion
Built using LangChain. 
Possible choice between three databases, two local (FAISS and chromadb) and one online (Vectara). 
<br>
Currently embeddings are done using OpenAI Embedding model.
<br>

#### To Do: <br>
* Do a better cleanup on the texts
* Experiment with other pdf loaders

## App
Simple HTML + Javascript frontend.<br>
FastAPI for the backend.<br>
Retrieval is done using LangChain RetrievalQAwithSources chain and OpenAI gpt-3.5-turbo model.

#### To Do: <br>
* Experiment with other chains<br>
* Use a customized prompt<br>

## Plans for the future: Going fully local
OpenAI is great but it's expensive. A solution for this would be to use a small local model. Mistral-7B is a good candidate for that. <br>
#### Step 1: Using ollama
First simple solution would be to plug the app to Mistral running loccally on ollama<br>
#### Step 2: Fine tuning Mistral
To get better results fine tuning may be the way!<br>
Building the fine tuning dataset can be done by asking any LLM to prepare a set of questions and answers for each documents.<br>
Then using it to fine tune a quantized version of Mistral-7B (or any other). That can then be run locally.<br>
