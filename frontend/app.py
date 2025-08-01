import streamlit as st
import requests
import json
import os
import time
from datetime import datetime

# Backend API URL
BASE_URL = "http://localhost:8000"

def display_quiz(topic: str, num_questions: int = 5):
    """Display interactive quiz with proper form submission and error handling"""
    try:
        with st.spinner("📝 Generating quiz..."):
            # Send request with error handling
            try:
                response = requests.post(
                    f"{BASE_URL}/quiz",
                    json={"topic": topic, "num_questions": num_questions},
                    timeout=30  # Add timeout
                )
                response.raise_for_status()  # Raise error for bad status codes
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
                st.error("Please make sure the backend server is running")
                return
            
            quiz_data = response.json()
            
            # Validate quiz data
            if "error" in quiz_data:
                st.error(f"❌ {quiz_data['error']}")
                return
            
            if not quiz_data.get("questions"):
                st.error("❌ No questions were generated. Please try again.")
                return
            
            # Display quiz header
            st.markdown(f"### 📝 {quiz_data['title']}")
            st.info(quiz_data['description'])
            
            # Create quiz form
            with st.form(key="quiz_form"):
                user_answers = {}
                
                # Display questions
                for i, question in enumerate(quiz_data['questions']):
                    st.markdown(f"### Question {i + 1}")
                    st.markdown(question['question'])
                    
                    # Create radio buttons for options
                    options = [(k, v) for k, v in question['options'].items()]
                    answer = st.radio(
                        "Select your answer:",
                        options,
                        format_func=lambda x: f"{x[0]}) {x[1]}",
                        key=f"q_{i}"
                    )
                    user_answers[i] = answer[0]
                    st.markdown("---")
                
                # Submit button
                submit_quiz = st.form_submit_button(
                    "Submit Quiz",
                    type="primary",
                    use_container_width=True
                )
                
                if submit_quiz:
                    st.session_state.quizzes_generated += 1
                    score = 0
                    total = len(quiz_data['questions'])
                    
                    # Display results
                    st.markdown("## 📊 Quiz Results")
                    
                    for i, question in enumerate(quiz_data['questions']):
                        is_correct = user_answers[i] == question['correct_answer']
                        if is_correct:
                            score += 1
                        
                        # Show feedback for each question
                        with st.expander(
                            f"Question {i + 1} {'✅' if is_correct else '❌'}"
                        ):
                            st.markdown(f"**Q:** {question['question']}")
                            st.markdown(f"**Your Answer:** {user_answers[i]}) {question['options'][user_answers[i]]}")
                            st.markdown(f"**Correct Answer:** {question['correct_answer']}) {question['options'][question['correct_answer']]}")
                            st.markdown("**Explanation:**")
                            st.info(question['explanation'])
                    
                    # Show final score
                    percentage = (score / total) * 100
                    st.success(f"### Your Score: {percentage:.1f}% ({score}/{total} correct)")
                    
                    if percentage >= 80:
                        st.balloons()
                    
                    # Update history
                    st.session_state.history.append({
                        "type": "Quiz",
                        "topic": topic,
                        "score": percentage,
                        "num_questions": total,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    save_history()
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")

# Streamlit page configuration
st.set_page_config(
    page_title="EduGuide: Study Helper",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "study_materials_generated" not in st.session_state:
    st.session_state.study_materials_generated = 0
if "quizzes_generated" not in st.session_state:
    st.session_state.quizzes_generated = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# Helper functions for history management
def save_history():
    """Save interaction history to a JSON file"""
    history_file = os.path.join("data", "history.json")
    os.makedirs("data", exist_ok=True)
    with open(history_file, "w", encoding='utf-8') as f:
        json.dump(st.session_state.history, f, indent=2, ensure_ascii=False)

def load_history():
    """Load interaction history from JSON file"""
    history_file = os.path.join("data", "history.json")
    if os.path.exists(history_file):
        with open(history_file, "r", encoding='utf-8') as f:
            st.session_state.history = json.load(f)

# Load history at startup
try:
    load_history()
except Exception as e:
    st.sidebar.warning(f"Could not load history: {str(e)}")

# Main title
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        📚 EduGuide: AI Study Assistant
    </h1>
    """, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("📍 Navigation")
    page = st.radio("Choose a feature", ["❓ Query", "📖 Study Material", "📝 Quiz"])

# File upload section - Moved below navigation
with st.sidebar:
    st.markdown("---")  # Add separator
    st.title("📄 Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=['pdf'])

    if uploaded_file is not None:
        try:
            with st.spinner("Processing PDF..."):
                # Save the uploaded file
                pdf_dir = os.path.join("data", "pdfs")
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, uploaded_file.name)
                
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Send PDF for processing
                with open(pdf_path, "rb") as f:
                    files = {"file": f}
                    response = requests.post(f"{BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.current_pdf = uploaded_file.name
                        st.sidebar.success(f"✅ Successfully processed: {uploaded_file.name}")
                    else:
                        st.sidebar.error("❌ Error processing PDF")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

    # Add clear PDF button
    if st.session_state.current_pdf:
        if st.sidebar.button("❌ Clear Current PDF"):
            st.session_state.current_pdf = None
            st.experimental_rerun()

# Main content area
if page == "❓ Query":  # Changed from 🔍 Search
    st.header("❓ Ask Questions")  # Changed from 💭 Search Documents
    if st.session_state.current_pdf:
        st.info(f"Currently loaded PDF: {st.session_state.current_pdf}")
    
    query = st.text_area("Enter your search query:", height=100)
    if st.button("🔍 Search", use_container_width=True):
        if query:
            with st.spinner("Searching and generating response..."):
                try:
                    response = requests.post(
                        f"{BASE_URL}/query",
                        json={"query": query}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display AI-generated answer
                        if "answer" in data:
                            st.markdown("### 🤖 AI Response")
                            st.markdown(data["answer"])
                            st.markdown("---")
                        
                        # Display source documents
                        if "sources" in data:
                            st.markdown("### 📚 Source Documents")
                            for i, source in enumerate(data["sources"], 1):
                                with st.expander(
                                    f"Source {i} from {source['source']} - Page {source['page']}"
                                ):
                                    st.markdown(source["content"])
                        
                        # Add to history
                        st.session_state.history.append({
                            "type": "Search",
                            "query": query,
                            "answer": data.get("answer", ""),
                            "sources": data.get("sources", []),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_history()
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please make sure the backend server is running")
        else:
            st.warning("⚠️ Please enter a search query")

elif page == "📖 Study Material":
    st.header("📖 Generate Study Material")
    if st.session_state.current_pdf:
        st.info(f"Currently loaded PDF: {st.session_state.current_pdf}")
    
    topic = st.text_input("Enter topic for study material:")
    if st.button("📚 Generate Study Material"):
        if topic:
            with st.spinner("Generating study material..."):
                try:
                    response = requests.post(
                        f"{BASE_URL}/study-material",
                        json={"topic": topic}
                    )
                    if response.status_code == 200:
                        st.session_state.study_materials_generated += 1
                        material = response.json()
                        
                        st.markdown("### 📖 Study Material")
                        st.markdown(f"**Topic:** {material['topic']}")
                        st.markdown(f"**Summary:**\n{material['summary']}")
                        with st.expander("Full Content"):
                            st.markdown(material['content'])
                        
                        # Add to history
                        st.session_state.history.append({
                            "type": "Study Material",
                            "topic": topic,
                            "summary": material['summary'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_history()
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a topic")

else:  # Quiz page
    st.header("📝 Generate Quiz")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Enter topic for quiz:", placeholder="e.g., Machine Learning")
    with col2:
        num_questions = st.number_input("Number of questions:", 1, 10, 5)
    
    if st.button("Generate Quiz", use_container_width=True):
        if topic:
            display_quiz(topic, num_questions)
        else:
            st.warning("⚠️ Please enter a topic")

# History Section in Sidebar
with st.sidebar:
    st.title("📜 History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"{item['type']} - {item['timestamp']}"):
                if item['type'] == "Search":
                    st.write(f"Query: {item['query']}")
                elif item['type'] == "Study Material":
                    st.write(f"Topic: {item['topic']}")
                    st.write(f"Summary: {item['summary'][:100]}...")
                else:  # Quiz
                    st.write(f"Topic: {item['topic']}")
                    st.write(f"Questions: {item['num_questions']}")
    else:
        st.info("No history yet")
