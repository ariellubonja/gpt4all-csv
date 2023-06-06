# This file was originally named db_loading.py in the Article
# Ariel - I've changed it significantly and renamed it to better match new functionality


from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings # Alpaca Embeddings
from langchain.vectorstores.faiss import FAISS # Vector similarity search


gpt4all_path = './models/gpt4all-converted.bin' 
llama_path = './models/ggml-model-q4_0.bin' 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# create the embedding object
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
    # encode_kwargs=encode_kwargs
)
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)


# Load our local index vector
# 1. You create embeddings of your document
# 2. Search for similarity betw. Query and index using FAISS
# 3. Give top-k similar vectors (their actual words) as Context to the model
index = FAISS.load_local("my_faiss_index_randal_papers", embeddings)


def run_query(query):
    """
        Take user query, find best embeddings with best context, print output
    """
    docs = index.similarity_search(query)
    # Get the matches best 3 results - defined in the function k=3
    print(f"\n\nThe question is: {query}")

    # print("\nHere is the most similar result, solely on Embedding similarity:")
    # print(docs[0].page_content)


    print("\n\n------------------------------------------------------------------")
    print("GPT4All Results, Using top K embeddings as context\n\n")


    # prompt template
    template = """
    Context: {context}
    ---
    Question: {question}
    Answer: """ # Let's NOT think step by step.


    matched_docs = index.similarity_search(query, k=5) 
    # Creating the context
    context = " ".join([doc.page_content for doc in matched_docs])
    # instantiating the prompt template and the GPT4All chain
    prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    llm_chain.run(query)



# test_query = ["What was the root cause of the damage to the dust cups?",
#     "Where is The Progesterone bulk product received from?",
#     "What was the root cause of the damage to the dust cups?"
# ]

# for q in test_query:
#     run_query(q)


while True:
    try:
        query = input("Your question (Ctrl + C to exit): ")

        run_query(query)
    except KeyboardInterrupt:
        print("\nLoop terminated by Ctrl + C")
        break
