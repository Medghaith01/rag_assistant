from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from tempfile import NamedTemporaryFile
from rag_model import (
    setup_llm,
    create_rag_chain,
    ask_question
)
from langchain_core.output_parsers import StrOutputParser
from auth import verify_api_key


app = FastAPI()

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_methods=[""],
    allow_headers=["*"],
)
"""
#In-memory chat history per session
chat_memory = {}

@app.post("/ask")
async def ask_endpoint(
    question: str = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(None),
    _: str = Depends(verify_api_key)  # Secure with API key

):
    try:
        # Initialize session memory if not exists
        if session_id not in chat_memory:
            chat_memory[session_id] = []

        history = chat_memory[session_id]

        # === CASE 1: PDF uploaded → use RAG ===
        if file:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                pdf_path = tmp.name

            qa_chain = create_rag_chain(pdf_path)
            response = ask_question(qa_chain, history, question)

            os.remove(pdf_path)  # Clean up temp file

        # === CASE 2: No PDF → plain LLM ===
        else:
            llm = setup_llm()
            prompt = f"""<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"""
            chain = llm | prompt | StrOutputParser()
            response = chain.invoke(question)
            history.extend([question, response])  # Store in history

        return JSONResponse({"response": response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset")
def reset_memory(session_id: str = Form(...)):
    chat_memory.pop(session_id, None)
    return {"message": f"Session {session_id} reset successfully"}