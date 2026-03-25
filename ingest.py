import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_PATH = "data/business_intelligence.pdf"
CHROMA_PATH = "chroma_db"


def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY saknas i .env-filen")

    embeddings = OpenAIEmbeddings(api_key=api_key)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    vectorstore.persist()


def main():
    print("Läser dokument...")
    documents = load_documents()

    print("Delar upp text...")
    chunks = split_text(documents)

    print("Skapar embeddings och sparar i Chroma...")
    create_vectorstore(chunks)

    print("Klart! Vektordatabasen är skapad.")


if __name__ == "__main__":
    main()