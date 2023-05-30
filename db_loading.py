
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS

# TEST FOR SIMILARITY SEARCH

# assign the path for the 2 models GPT4All and Alpaca for the embeddings 
gpt4all_path = './models/gpt4all-converted.bin' 
llama_path = './models/ggml-model-q4_0.bin' 
# Calback manager for handling the calls with  the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# create the embedding object
embeddings = LlamaCppEmbeddings(model_path=llama_path)
# create the GPT4All llm object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)



# Load our local index vector db
index = FAISS.load_local("my_faiss_index", embeddings)
# Hardcoded question
query = "What was the root cause of the damage to the dust cups?"
docs = index.similarity_search(query)
# Get the matches best 3 results - defined in the function k=3
print(f"\n\nThe question is: {query}")

print("\nHere the result of the semantic search on the index, without GPT4All:")
print(docs[0].page_content)


print("\n\n------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")


print("\n\nGPT4All Results\n\n")


# create the prompt template
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

# Hardcoded question
# matched_docs, sources = similarity_search(query, index)
matched_docs = index.similarity_search(query, k=3) 
# Creating the context
context = " ".join([doc.page_content for doc in matched_docs])
# instantiating the prompt template and the GPT4All chain
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)

llm_chain.run(query)

while True:
    try:
        query = input("Your question (Ctrl + C to exit): ")

        # Print the result
        llm_chain.run(query)
    except KeyboardInterrupt:
        print("\nLoop terminated by Ctrl + C")
        break

    
# print(llm_chain.run(query))
