from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAG
import os
import shutil
import uvicorn
import logging
from preprocess import preprocess_pdfs
from embeddings import create_embeddings
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EduGuide Backend")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag = RAG()

# Request models
class QueryRequest(BaseModel):
    query: str

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5

class StudyMaterialRequest(BaseModel):
    topic: str

class QuizAnswer(BaseModel):
    question_id: int
    selected_answer: str

class QuizSubmission(BaseModel):
    quiz_id: str
    answers: List[QuizAnswer]

class QuizFeedback(BaseModel):
    question_id: int
    question: str
    your_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str

class QuizResult(BaseModel):
    quiz_id: str
    score: float
    total_questions: int
    correct_answers: int
    feedback: List[QuizFeedback]

query_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Based on the following context, provide a comprehensive answer to: {query}

    Context:
    {context}

    Instructions:
    1. Base your answer ONLY on the provided context
    2. Use markdown formatting in your response
    3. Structure your answer with:
       - Main explanation
       - Key points
       - Examples (if available)
    4. If the context doesn't contain sufficient information, clearly state that

    Answer:"""
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Set up directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    pdfs_dir = os.path.join(data_dir, "pdfs")
    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    
    # Create necessary directories
    os.makedirs(pdfs_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)

    try:
        # First save to uploads directory (temporary storage)
        temp_file_path = os.path.join(uploads_dir, file.filename)
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Then copy to pdfs directory (permanent storage)
        final_file_path = os.path.join(pdfs_dir, file.filename)
        shutil.copy2(temp_file_path, final_file_path)
        logger.info(f"Saved PDF to {final_file_path}")

        # Preprocess PDFs and generate embeddings
        logger.info("Preprocessing PDFs...")
        documents = preprocess_pdfs()
        
        logger.info("Generating embeddings...")
        vectorstore = create_embeddings(documents=documents)
        
        if vectorstore:
            # Reload RAG system with new embeddings
            rag.load_index()
            return {"message": f"File {file.filename} uploaded and processed successfully"}
        else:
            return {"message": f"File {file.filename} uploaded but no content was extracted for processing"}
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system and generate an LLM response"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Search for relevant documents
        docs = rag.search_documents(request.query, k=5)
        if not docs:
            logger.warning("No documents found")
            return {
                "answer": "No relevant information found in the documents.",
                "sources": []
            }
        
        # Format context with source information
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown').split('/')[-1]  # Get just filename
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content.strip()
            context_parts.append(f"[Source: {source}, Page {page}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        try:
            logger.info("Generating LLM response...")
            # Generate LLM response with formatted context
            response = rag.llm.invoke(
                query_prompt.format(
                    query=request.query,
                    context=context
                )
            )
            
            if not hasattr(response, 'content'):
                raise ValueError("No content in LLM response")
            
            logger.info("LLM response generated successfully")
            
            # Structure the response
            formatted_response = {
                "answer": response.content,
                "sources": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown").split('/')[-1],
                        "page": doc.metadata.get("page", 0)
                    } for doc in docs
                ],
                "query": request.query
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            return {
                "answer": "Found relevant information but couldn't generate a response. Here are the relevant excerpts:",
                "sources": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "source": doc.metadata.get("source", "Unknown").split('/')[-1],
                        "page": doc.metadata.get("page", 0)
                    } for doc in docs
                ],
                "query": request.query,
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Query Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/study-material")
async def generate_study_material(request: StudyMaterialRequest):
    """Generate study material for a topic"""
    try:
        material = rag.generate_study_material(request.topic)
        if "error" in material:
            raise HTTPException(status_code=404, detail=material["error"])
        return material
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating study material: {str(e)}")

@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """Generate quiz questions"""
    try:
        quiz = rag.generate_quiz(request.topic, request.num_questions)
        if "error" in quiz:
            raise HTTPException(status_code=404, detail=quiz["error"])
        
        # Add quiz_id and timestamp
        quiz["quiz_id"] = f"quiz_{int(time.time())}"
        quiz["timestamp"] = datetime.now().isoformat()
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.post("/submit-quiz", response_model=QuizResult)
async def submit_quiz(submission: QuizSubmission):
    """Grade quiz submission and provide feedback"""
    try:
        quiz = rag.get_quiz(submission.quiz_id)
        
        results = {
            "quiz_id": submission.quiz_id,
            "score": 0,
            "total_questions": len(quiz["questions"]),
            "correct_answers": 0,
            "feedback": []
        }
        
        for answer in submission.answers:
            question = quiz["questions"][answer.question_id]
            is_correct = answer.selected_answer == question["correct_answer"]
            
            feedback = QuizFeedback(
                question_id=answer.question_id,
                question=question["question"],
                your_answer=answer.selected_answer,
                correct_answer=question["correct_answer"],
                is_correct=is_correct,
                explanation=question.get("explanation", "No explanation available")
            )
            
            if is_correct:
                results["correct_answers"] += 1
            
            results["feedback"].append(feedback)
        
        results["score"] = (results["correct_answers"] / results["total_questions"]) * 100
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error grading quiz: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Changed to use import string
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )