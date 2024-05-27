from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from transformers import pipeline
from typing import List

app = FastAPI()

# Load a QA pipeline using transformers
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    # Splitting text into chunks of specified size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    text = extract_text_from_pdf(file)
    chunks = split_text_into_chunks(text)
    return {"filename": file.filename, "chunks": chunks}

@app.post("/chat/")
async def chat_with_pdf(query: str, chunks: List[str]):
    # Simple context creation by concatenating all chunks
    context = " ".join(chunks)
    
    # Use the QA pipeline to answer the question based on the context
    result = qa_pipeline(question=query, context=context)
    return {"question": query, "answer": result['answer']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
