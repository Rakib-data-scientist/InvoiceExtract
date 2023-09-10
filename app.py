import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def upload_file():
    """Function to upload a file and return its path."""
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name
    return None

def load_pdf_content(file_path):
    """Function to load the content of a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()

def extract_entities(page_content):
    """Function to extract entities from the page content using the LLM model."""
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin", 
        model_type="llama", 
        config={'max_new_tokens': 128, 'temperature': 0.01}
    )
    
    template = """
    Extract Invoice Number, Order Number, Invoice Date,Due date, Total Due,Service,
    Rate/Price, name of organization, address, date, 
    Qty, Tax, Amount {page_content}
    Output: entity : type
    """
    prompt_template = PromptTemplate(input_variables=["page_content"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain.run(page_content=page_content)

def display_content(pages):
    """Function to display the content of the uploaded file."""
    st.write(f"Number of pages: {len(pages)}")
    for page in pages:
        st.write(page.page_content)

def display_entities(entities):
    """Function to display the extracted entities in a table."""
    st.write("Extracted entities:")
    table_data = [line.split(":") for line in entities]
    st.table(table_data)

def main():
    st.title("Invoice Entity Extractor")

    temp_file_path = upload_file()

    if temp_file_path:
        pages = load_pdf_content(temp_file_path)
        display_content(pages)

        if pages:
            result = extract_entities(pages[0].page_content)
            if result:
                entities = result.strip().split("\n")
                display_entities(entities)
            else:
                st.write("No entities extracted.")
        else:
            st.write("No pages found in the uploaded file.")
    else:
        st.write("No file uploaded.")

if __name__ == "__main__":
    main()
