import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    """Retrieval-Augmented Generation system for educational content."""
    
    def __init__(self):
        """Initialize RAG system with necessary paths and models."""
        # Set up paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.index_dir = os.path.join(self.data_dir, "index")
        self.history_file = os.path.join(self.data_dir, "history.json")

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load existing index
        self.vectorstore = self._load_vectorstore()
        logger.info("RAG system initialized")

        # Initialize Gemini model with correct configuration
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Updated to stable version
            temperature=0.7,
            convert_system_message_to_human=True,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=2048,
            top_p=0.95,
            top_k=40
        )

        # Define prompts
        self.study_prompt = PromptTemplate(
            input_variables=["context", "topic"],
            template="""You are an expert educational content creator. Create a comprehensive study guide about {topic} using this context:

            Context:
            {context}

            Structure your response exactly as follows:
            1. Summary (2-3 clear, concise sentences)
            2. Key Concepts (bullet points of main ideas)
            3. Detailed Explanations (step-by-step breakdown)
            4. Examples (real-world applications)
            5. Review Points (key takeaways)

            Focus on accuracy and clarity."""
        )

        self.quiz_prompt = PromptTemplate(
            input_variables=["context", "topic", "num_questions"],
            template="""You are an expert assessment creator. Generate {num_questions} multiple-choice questions about {topic} based on:

            Context:
            {context}

            Format each question exactly as:
            Question X: [Clear, specific question]
            A) [Option]
            B) [Option]
            C) [Option]
            D) [Option]
            Correct Answer: [Letter]

            Ensure questions test understanding, not just recall."""
        )

        self.active_quizzes = {}  # Store active quizzes in memory

    def _load_vectorstore(self) -> Optional[FAISS]:
        """Load existing FAISS index"""
        try:
            if os.path.exists(self.index_dir):
                logger.info("Loading existing FAISS index...")
                return FAISS.load_local(
                    self.index_dir, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            logger.warning("No existing index found")
            return None
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return None
            
    def load_index(self) -> None:
        """Reload the FAISS index"""
        logger.info("Reloading FAISS index...")
        self.vectorstore = self._load_vectorstore()
        if self.vectorstore:
            logger.info("FAISS index reloaded successfully")
        else:
            logger.warning("Failed to reload FAISS index")

    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        if not self.vectorstore:
            logger.warning("No vectorstore available")
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                logger.warning(f"No documents found for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def generate_study_material(self, topic: str) -> Dict[str, Any]:
        """Generate study material using Gemini"""
        docs = self.search_documents(topic, k=5)
        if not docs:
            return {"error": "No relevant content found"}

        try:
            context = "\n".join([doc.page_content for doc in docs])
            response = self.llm.invoke(
                self.study_prompt.format(
                    context=context,
                    topic=topic
                )
            )

            study_material = {
                "topic": topic,
                "content": response.content,
                "summary": response.content.split("\n")[0],
                "generated_at": str(datetime.now())
            }
            
            self._log_history({
                "type": "study_material",
                "topic": topic,
                "timestamp": str(datetime.now())
            })
            
            return study_material
        except Exception as e:
            logger.error(f"Error generating study material: {str(e)}")
            return {"error": str(e)}

    def get_quiz(self, quiz_id: str) -> Dict[str, Any]:
        """Retrieve a quiz by ID"""
        if quiz_id not in self.active_quizzes:
            raise ValueError(f"Quiz {quiz_id} not found")
        return self.active_quizzes[quiz_id]

    def generate_quiz(self, topic: str, num_questions: int = 3) -> Dict[str, Any]:
        """Generate quiz using Gemini"""
        try:
            docs = self.search_documents(topic)
            if not docs:
                return {"error": "No relevant content found"}

            context = "\n".join([doc.page_content for doc in docs])
            
            # Updated prompt with simpler structure
            quiz_prompt = f"""Create a quiz about {topic} with exactly {num_questions} multiple-choice questions.
            Use this context: {context}

            IMPORTANT: Return ONLY valid JSON in this exact format:
            {{
                "title": "Quiz on {topic}",
                "description": "Test your knowledge of {topic}",
                "questions": [
                    {{
                        "id": 1,
                        "question": "Write your question here?",
                        "options": {{
                            "A": "First option",
                            "B": "Second option",
                            "C": "Third option",
                            "D": "Fourth option"
                        }},
                        "correct_answer": "A",
                        "explanation": "Why this answer is correct"
                    }}
                ]
            }}"""

            response = self.llm.invoke(quiz_prompt)
            response_text = response.content.strip()

            # Clean and parse response
            try:
                if "```" in response_text:
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()

                quiz_content = json.loads(response_text)
                quiz_id = f"quiz_{int(time.time())}"

                # Create quiz structure
                quiz = {
                    "quiz_id": quiz_id,
                    "topic": topic,
                    "title": quiz_content.get("title", f"Quiz on {topic}"),
                    "description": quiz_content.get("description", "Test your knowledge!"),
                    "questions": [],
                    "total_points": 0,
                    "timestamp": datetime.now().isoformat()
                }

                # Process questions
                if "questions" in quiz_content and isinstance(quiz_content["questions"], list):
                    for i, q in enumerate(quiz_content["questions"]):
                        if self._validate_question(q):
                            question = {
                                "id": i,
                                "question": q["question"].strip(),
                                "options": {k: v.strip() for k, v in q["options"].items()},
                                "correct_answer": q["correct_answer"].strip(),
                                "explanation": q["explanation"].strip(),
                                "points": 5
                            }
                            quiz["questions"].append(question)
                            quiz["total_points"] += question["points"]

                if not quiz["questions"]:
                    return {"error": "No valid questions generated"}

                # Store quiz
                self.active_quizzes[quiz_id] = quiz
                return quiz

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}\nResponse: {response_text[:200]}...")
                return {"error": "Failed to parse quiz response"}

        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return {"error": str(e)}

    def _validate_question(self, question: Dict) -> bool:
        """Validate quiz question format"""
        required_fields = ["question", "options", "correct_answer", "explanation"]
        if not all(field in question for field in required_fields):
            return False
        
        if not isinstance(question["options"], dict) or \
           not all(key in question["options"] for key in ["A", "B", "C", "D"]):
            return False
        
        if question["correct_answer"] not in ["A", "B", "C", "D"]:
            return False
        
        return True

    def _log_history(self, entry: Dict[str, str]) -> None:
        """Log interactions to history file"""
        try:
            if not isinstance(entry, dict):
                logger.error("Invalid history entry format")
                return
                
            history = []
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Validate and load existing history
            if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
                try:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            logger.warning("Invalid history format, starting fresh")
                            history = []
                except json.JSONDecodeError:
                    logger.warning("Corrupted history file, starting fresh")
                    history = []
                    
            # Add new entry with timestamp
            entry["timestamp"] = str(datetime.now())
            history.append(entry)
            
            # Save updated history
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error logging history: {str(e)}")


def main():
    """Test the RAG system functionality"""
    rag = RAG()
    
    print("\n=== Testing Document Search ===")
    test_query = "What is machine learning?"
    docs = rag.search_documents(test_query)
    print(f"\nSearch results for: {test_query}")
    if docs:
        for i, doc in enumerate(docs, 1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Page: {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:200]}...")
    else:
        print("No results found")

    print("\n=== Testing Study Material Generation ===")
    study_material = rag.generate_study_material("machine learning basics")
    if isinstance(study_material, dict) and "error" not in study_material:
        print("\nGenerated Study Material:")
        print(f"Topic: {study_material['topic']}")
        print(f"Summary: {study_material['summary']}")
        print(f"Content Preview: {study_material['content'][:200]}...")
    else:
        print(f"Error: {study_material.get('error', 'Unknown error')}")

    print("\n=== Testing Quiz Generation ===")
    quiz = rag.generate_quiz("machine learning basics")
    if "error" not in quiz:
        print("\nGenerated Quiz:")
        print(f"Topic: {quiz['topic']}")
        for i, q in enumerate(quiz['questions'], 1):
            print(f"\nQuestion {i}: {q['question']}")
            print("Options:", ", ".join(q['options']))
            print(f"Correct Answer: {q['correct_answer']}")
    else:
        print(f"Error: {quiz['error']}")


if __name__ == "__main__":
    main()