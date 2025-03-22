from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF

from dbUtils import DBUtils
import bs4
from langchain_community.document_loaders import WebBaseLoader

def extract_text_from_pdf(pdf_path):
    text = ""
    phases = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            for lines in page.get_text().split('\n'):
                line = lines.strip()
                if (' ' in line):
                    print("skip line")
                elif len(line) > 1:
                
                    phases.append(lines.strip())
             
            
 

    return phases

def calculate_chunk_ids(chunks):

    # ID => Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        print("find chuck "+current_page_id)
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def split_documents(documents: list[Document]):
       
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=80

    )
    return text_splitter.split_documents(documents)

def indexing_to_chroma(chunks: list[Document], indexing_to_chroma_path=DBUtils.CHROMA_PATH, model_type="m3e"):
    # Load the existing database.
    utils = DBUtils()

    db = utils.get_chroma(indexing_to_chroma_path, model_type )
  

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
  
    existing_sources = set(existing_items["ids"])
  
    print(f"Number of existing documents in DB: {len(existing_sources)}")
    print(existing_sources)

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_sources:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_sources = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_sources)
     

    else:
        print("✅ No new documents to add")

def pdf_content_into_documents (pdf_path: str,  indexing_to_chroma_path=DBUtils.CHROMA_PATH, model_type="m3e") -> list[Document]:
    """Load and process PDF content into document chunks."""
    # Load PDF
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()
    docs = split_documents(documents)
    indexing_to_chroma(docs, indexing_to_chroma_path, model_type)
    return docs

def format_docs(docs):
    """Format documents for display."""
    if isinstance(docs[0], tuple):
        return "\n\n---\n\n".join([doc[0].page_content for doc in docs])
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def web_content_into_documents(url: str, model_type="m3e") -> list[Document]:
    loader = WebBaseLoader(url, bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),)
    return loader.load()