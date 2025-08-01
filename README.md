<<<<<<< HEAD
# RAG-Powered-StudyBuddy
=======
# EduGuide: AI-Powered Study Assistant

An intelligent study companion that helps students learn from PDF documents through interactive features and AI-powered assistance.

## Features

- **Document Processing**: Upload and process PDF documents
- **Interactive Q&A**: Ask questions about your documents
- **Study Material Generation**: Create comprehensive study guides
- **Quiz Generation**: Generate interactive quizzes with explanations
- **Progress Tracking**: Track your learning progress

## Project Structure

```
EduGuide/
├── backend/           # FastAPI backend
├── frontend/         # Streamlit frontend
├── data/            # Data storage (gitignored)
└── requirements.txt  # Project dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/asiifkarim/RAG-Powered-StudyBuddy.git
cd RAG-Powered-StudyBuddy
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file and add:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

6. Start the frontend (in a new terminal):
```bash
cd frontend
streamlit run app.py
```

## Usage

1. Upload a PDF document using the sidebar
2. Ask questions about the content
3. Generate study materials
4. Create and take quizzes
5. Track your progress

## Technologies Used

- Frontend: Streamlit
- Backend: FastAPI
- AI: LangChain, Google Gemini
- Vector Store: FAISS
- Document Processing: PyPDF2
>>>>>>> f855ef0 (Initial commit with backend and frontend)
