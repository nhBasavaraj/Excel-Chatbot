import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_UB7WOXdrwkVVsSrLa44uB51Md4dZcTI4SbXDe"

# Initialize Pinecone
pinecone.init(api_key='7352079a-ceea-4e89-8b2a-e1ceaeec0287', environment='gcp-starter')

# Load and preprocess the CSV document with explicit encoding (e.g., 'utf-8', 'latin-1')
loader = CSVLoader('WORK/files/Facebook_data_V3.csv', encoding='Windows-1252')  # Specify the correct encoding
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"])
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

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

# Start chatting with the chatbot
chat_history = []
while True:
    query = input('\n Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    #print("\n", result)
    print('\n Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
