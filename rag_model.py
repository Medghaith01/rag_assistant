import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#langchain_cummunity imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv



load_dotenv()  # loads variables from .env file
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
print(HF_TOKEN)


MODEL_NAME = "google/gemma-2-2b-it"
pdf_path = "./temp_uploads/1da6c989-8be8-45f3-96ce-710c5e56eb58.pdf"


def setup_llm(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)

def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return FAISS.from_documents(chunks, embeddings)

def create_prompt_template():
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""<bos><start_of_turn>user
You are an expert research assistant to a university professor. You have access to research documents and should help with academic inquiries in a scholarly and professional manner.

Context from research documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}
<end_of_turn>
<start_of_turn>model
"""
    )
def format_chat_history(history):
    if not history:
        return "No previous conversation."
    result = []
    for i in range(0, len(history), 2):
        if i+1 < len(history):
            result.append(f"Human: {history[i]}")
            result.append(f"Assistant: {history[i+1]}")
    return "\n".join(result)

def add_to_memory(history, question, answer, max_exchanges=6):
    history.extend([question, answer])
    max_items = max_exchanges * 2
    if len(history) > max_items:
        del history[:-max_items]
    return history


llm = setup_llm()
chunks = load_and_process_pdf(pdf_path)
vectorstore = create_vector_store(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = create_prompt_template()

def create_rag_chain(model=llm,rt=retriever,pt=prompt_template):

    def format_docs(docs):
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    qa_chain = (
        RunnableMap({
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: format_docs(rt.get_relevant_documents(x["question"]))
        })
        | pt
        | model
        | StrOutputParser()
    )
    return qa_chain

def ask_question(qa_chain):
    history = []
    while True:
        question = input("You: ")
        if question.strip().lower() == "q":
            print("Exiting.")
            break

        chat_history = format_chat_history(history)
        response = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print(f"Assistant: {response}")
        add_to_memory(history, question, response)

#testing
qa_chain = create_rag_chain(llm,retriever,prompt_template)

