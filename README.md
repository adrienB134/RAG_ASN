# RAG_ASN
## 🚧 Building a full RAG chatbot 🚧

The goal is to build a full Retrieval Augmented Generation chatbot on french data from the nuclear safety agency.
I first completed the following steps:<br>
 * Getting the data<br>
 * Processing it and storing it in a vector store<br>
 * Build a Chat app doing the retrieval and displaying it.<br>
  
I got decent results using OpenAI embeddings and GPT4, but I wanted something that could be used without sending data to a third party as data privacy is a huge concern for companies.<br>
So I switched to open-source models running on private inference endpoints, first with Mistral then later on with Mixtral when it came out. Mistral results where poor and Mixtral a bit better but not great, but the main issue was the downgrade in retrieved documents relevance probably due to the embedding model. <br><br>
To improve the results I'm exploring the following ideas:<br>
- Use an embedding model trained on french (first results on a small sample dataset are looking good)<br>
- Finetune Mistral (only one I can do on colab) on my data to gain domain knowledge and then use the HyDE retrieval technique (short: ask the llm to imagine the answer document then do a similarity search with it)<br>
- Research tends to show that late-interaction models like ColBERT do better for specific domain RAG, and the architecture will solve my embeddings problem

## Getting the data
Scraped the ASN site to get the publicly available pdfs regarding nuclear site inspections.
Done with Scrapy, see scraping folder.

## Ingestion
The ingestion script does a few things:<br>
- Using unstuctured.io package, it extracts and cleans the raw text from the pdfs<br>
- It then splits it into chunks<br>
- Embeds those chunks using an embeddings model from huggingface<br>
- Finally it stores the embeddings in a local chromadb vectorstore<br>

<br>
Currently embeddings are done using WhereIsAI/UAE-Large-V1. 
Since it's trained for the english language, it might not be optimal, I need to do more research.
<br>


## App
Simple HTML + Javascript frontend.<br>
FastAPI for the backend.<br>
Retrieval is done using LangChain and the LLM is Mistral.

#### To Do: <br>
* Introduce a reranking step. See [here](https://medium.com/llamaindex-blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)<br>
* Pimp out the frontend


## Finetuning for domain knowledge and better results
#### Mistral: <br>
Latest dataset uploaded to [HF](https://huggingface.co/datasets/AdrienB134/ASN_pairs). <br>
Need to put more thought in how to structure the dataset to get what I want.  <br>
#### ColBERT: <br>
Retrieval without finetuning looks good. <br>
A lot of issues with accents when creating dataset for fine-tuning, I need to automate the cleaning or find a way to get the queries without accent. <br>


## Plans for the future: 
Do something like this to better evaluate model performance https://github.com/Arize-ai/LLMTest_NeedleInAHaystack. 
