import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()


def load_pdf(path):
    print(f"Loading {path}...")
    loader = PyPDFLoader(path)
    return loader.load()


def generate_context_txt(pages):
    print("Generating context txt file...")
    with open("docs.txt", "w", encoding="utf-8") as f:
        for p in pages:
            f.write(p.page_content + "\n\n")
    return open("docs.txt", "r", encoding="utf-8").read()


def split_embedded_data(docs):
    print("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=75, add_start_index=True
    )
    texts = splitter.split_text(docs)
    print("Embedding data...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_dim = len(embeddings.embed_query("test"))
    print("Creating index...")
    index = faiss.IndexFlatIP(embedding_dim)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("Adding texts...")
    store.add_texts(texts)
    print("Saving index...")
    if not os.path.exists("./faiss_index"):
        print("Saving index...")
        store.save_local("./faiss_index/")
    else:
        print("Loading index...")
        store = FAISS.load_local(
            "./faiss_index/", embeddings, allow_dangerous_deserialization=True
        )
    return store


app = FastAPI()


@app.get("/")
def root():
    return {
        "Welcome to the LLM Test API! This API allows you to ask questions about the PDF document 'LLM_Test.pdf'.\n\nTo use this API, simply make a GET request to '/query/<your_question>' where <your_question> is your question or topic of interest.\n\nExample usage: http://localhost:8000/query/What%20is%LLM?\n\nThis will return the answer to your question based on information extracted from the PDF."
    }


@app.get("/ask/{question}")
def query(question: str):
    path = r"LLM_Test.pdf"
    pages = load_pdf(path)
    context_txt = generate_context_txt(pages)
    vector_store = split_embedded_data(context_txt)

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest):
        last_query = request.state["messages"][-1].text
        retreived_docs = vector_store.similarity_search(last_query, k=3)

        docs = "\n".join([doc.page_content for doc in retreived_docs])

        return (
            f"You are an AI assistant.\nUse ONLY the following context:\n\n{docs}\n\n"
        )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    agent = create_agent(model, middleware=[prompt_with_context])

    print("Query:", question)

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]}, stream_node="messages"
    ):
        if "model" in step and "messages" in step["model"]:
            final_ans = ""
            msg = step["model"]["messages"][-1]
            final_ans = msg.content
    return {"message": final_ans}
