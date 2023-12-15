import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_VAU7M3GCFy0dkrZQqOKxVh1u7fsUQjW1vrdhO"

# Initialize Pinecone
pinecone.init(api_key='7352079a-ceea-4e89-8b2a-e1ceaeec0287', environment='gcp-starter')

# Load and preprocess the PDF document
loader = PyPDFLoader('WORK/files/csv-pdf-Facebook.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
#print(embeddings)

# Set up the Pinecone vector database
index_name = "llama-2-csv"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)


# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)


# Start chatting with the chatbot with history
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
    

# # Start chatting with the chatbot without history
# import os
# import sys
# import pinecone
# from langchain.llms import Replicate
# from langchain.vectorstores import Pinecone
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain

# # Replicate API token
# os.environ['REPLICATE_API_TOKEN'] = "r8_VAU7M3GCFy0dkrZQqOKxVh1u7fsUQjW1vrdhO"

# # Initialize Pinecone
# pinecone.init(api_key='7352079a-ceea-4e89-8b2a-e1ceaeec0287', environment='gcp-starter')

# # Load and preprocess the PDF document
# loader = PyPDFLoader('WORK/files/xlsx-txt-employee-sample-data.pdf')
# documents = loader.load()

# # Split the documents into smaller chunks for processing
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings()

# # Set up the Pinecone vector database
# index_name = "llama-2-excel"
# index = pinecone.Index(index_name)
# vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# # Initialize Replicate Llama2 Model
# llm = Replicate(
#     model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
#     input={"temperature": 0.75, "max_length": 3000}
# )

# # Set up the Conversational Retrieval Chain
# qa_chain = ConversationalRetrievalChain(
#     llm=llm,
#     retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=False,
#     model_kwargs={},
# )



# # Start chatting with the chatbot
# while True:
#     query = input('Prompt: ')
#     if query.lower() in ["exit", "quit", "q"]:
#         print('Exiting')
#         sys.exit()
#     result = qa_chain({'question': query})
#     print('Answer:', result['answer'], '\n')

    