# RAG_ASN
## 🚧 Building a full RAG chatbot 🚧

The goal is to build a full Retrieval Augmented Generation chatbot on french data from the nuclear safety agency.
I first completed a basic dense retrieval MVP, which included the following steps:<br>
 * Getting the data<br>
 * Processing it and storing it in a vector store<br>
 * Building a Chat app doing the retrieval and displaying it.  
   
Now my end goal is to turn this into a more complex product where the model has domain knowledge and is not only doing basic retrieval.<br>
## Models
I got decent results using OpenAI embeddings and GPT4, but I wanted something that could be used without sending data to a third party as data privacy is a huge concern for companies.
So I switched to open-source models running on private inference endpoints, first with Mistral then later on with Mixtral when it came out.
Did it work? No! <br><br>
Mistral results where poor and Mixtral a bit better but not great, the main issue was the downgrade in retrieved documents relevance due to the embedding model. The mistake I made was using the UAE-Large-V1 for embeddings which is only trained on the english language. I switched to biencoder-camembert from [antoine louis](https://huggingface.co/antoinelouis) (he is doing excellent work training models for the french language), results were much better. 
To improve the results I'm exploring the following ideas:<br>
- Use the [HyDE](https://arxiv.org/abs/2212.10496) retrieval technique (short: ask the llm to imagine the answer document then do a similarity search with it)<br>
- Research tends to show that late-interaction models like ColBERT do better for specific domain RAG than single vector representation  retrievers
- I also need to build a more robust evaluation method, as I find asking the same set of questions and comparing the results a tad too empirical for my liking

## Getting the data
Scraped the ASN site to get the publicly available pdfs regarding nuclear site inspections.
Done with Scrapy, see scraping folder.

## Ingestion
The ingestion script does a few things:<br>
- Using unstuctured.io package, it extracts and cleans the raw text from the pdfs<br>
- It then splits it into chunks<br>
- Embeds those chunks using an embeddings model from huggingface<br>
- Finally it stores the embeddings in a local chromadb vectorstore<br>

## App
Simple HTML + Javascript frontend.<br>
FastAPI for the backend.<br>
Retrieval is done through LangChain (similarity search on the vector database) and the LLM used to format the answer is whichever one we deploy to our endpoint.

#### To Do: <br>
* Introduce a reranking step. See [here](https://medium.com/llamaindex-blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)<br>
* Pimp out the frontend

#### Things I need to look into: 
DSPy: https://github.com/stanfordnlp/dspy<br>
DALM: https://github.com/arcee-ai/DALM

## Plans for the future: 
Do something like this to better evaluate model performance https://github.com/Arize-ai/LLMTest_NeedleInAHaystack. 



