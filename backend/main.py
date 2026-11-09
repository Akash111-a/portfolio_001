import os
import json
from typing import Generator, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "Mistral Medium 3.5 (128B)"

app = FastAPI(title="Abhisek Jena Portfolio AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Consolidated factual knowledge base from all resumes
PROFILE_DATA = {
    "personal": {
        "full_name": "Abhisek Jena",
        "preferred_name": "Akash",
        "location": "Bhadrak, Odisha - 756119, India",
        "phone": "(+91) 7656817298",
        "email": "abhisekakash111@gmail.com",
        "github": "https://github.com/Akash111-a",
        "title": "Computer Science Engineer | QA & SDE Specialist",
        "languages": ["English", "Hindi", "Odia"],
        "hobbies": ["Competitive Gaming", "Playing Cricket"]
    },
    "education": [
        {
            "degree": "B.Tech in Computer Science and Engineering",
            "institution": "Balasore College of Engineering and Technology",
            "timeline": "2022 - 2026",
            "score": "CGPA: 8.01 / 10"
        },
        {
            "degree": "Higher Secondary Education (Class XII - Science)",
            "institution": "Bhadrak Autonomous College",
            "timeline": "2019 - 2021",
            "score": "Percentage: 74%"
        },
        {
            "degree": "Secondary Education (Class X)",
            "institution": "Nadigaon High School",
            "timeline": "2019",
            "score": "Percentage: 70%"
        }
    ],
    "skills": {
        "manual_testing": ["Functional", "Integration", "System", "Regression", "Smoke", "Sanity Testing", "BVA", "Equivalence Partitioning", "SDLC", "STLC", "Defect Life Cycle"],
        "databases_sql": ["Oracle SQL*Plus (400-500 practiced queries)", "MySQL", "SQL Server", "Joins", "Correlated Subqueries", "GROUP BY/HAVING", "Views", "Pseudo Columns"],
        "programming": ["Python", "Java", "C++", "SQL", "JavaScript"],
        "devops_cloud": ["Docker", "Kubernetes", "AWS (EC2, IAM, S3, VPC)", "Jenkins", "Ansible", "Git", "GitHub", "Linux"],
        "ai_ml": ["PyTorch Geometric", "OpenCV", "MediaPipe", "Pandas", "NumPy", "Speech Recognition", "FastAPI"]
    },
    "projects": [
        {
            "title": "DoomAI — Real-Time Resume Assistant",
            "stack": "FastAPI, Groq Cloud API, Llama-3.3, PyPDF",
            "github": "https://github.com/Akash111-a/DOOM_AI",
            "impact": "Interactive candidate intelligence platform streaming contextual evaluations directly to recruiters."
        },
        {
            "title": "JARVIS AI Voice Assistant with Gesture Control",
            "stack": "Python, PyQt6, OpenCV, MediaPipe, Speech Recognition",
            "impact": "Touchless interface enabling real-time hand-tracking mouse control and NLP voice automation."
        },
        {
            "title": "ShoppersStack E-Commerce Manual Testing Suite",
            "stack": "Manual Testing, STLC, Test Scenarios, BVA, GitHub",
            "github": "https://github.com/Akash111-a/SHOPPERSTACK_TEST_CASE",
            "impact": "Comprehensive test case design, positive/negative boundary validation, and defect tracking across e-commerce flows."
        },
        {
            "title": "Graph Neural Networks for Detecting Transactional Fraud",
            "stack": "Python, PyTorch Geometric, Pandas, NumPy, GNN",
            "impact": "Achieved 97.4% accuracy on IEEE-CIS Fraud Detection dataset with custom graph topological pipeline."
        },
        {
            "title": "Enterprise Database & Query Optimization",
            "stack": "Oracle SQL*Plus, Relational Schemas, Views, Correlated Subqueries",
            "impact": "Practiced and benchmarked 400+ complex relational queries on employee/department datasets."
        }
    ]
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

@app.get("/")
def home():
    return {"status": "online", "engineer": "Abhisek Jena (Akash)"}

def generate_ai_stream(history: List[ChatMessage]) -> Generator[str, None, None]:
    system_prompt = f"""
You are the official Personal AI Assistant for Abhisek Jena (Akash).
Answer user and recruiter questions accurately using this factual profile data:
{json.dumps(PROFILE_DATA, indent=2)}

Guidelines:
1. Speak professionally, concisely, and warmly.
2. Always answer based only on Abhisek's documented skills, projects, and academics. Never fabricate companies, degrees, or experience.
3. If information is not in his profile, say: "I don't have that specific detail in Abhisek's verified profile. You can reach him directly at abhisekakash111@gmail.com."
4. Use neat Markdown (bullets, bold text) for readability.
"""
    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-6:]:
        messages.append({"role": m.role, "content": m.content})

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[AI Service Error: {str(e)}]"

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    if not data.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    return StreamingResponse(generate_ai_stream(data.messages), media_type="text/plain; charset=utf-8")

@app.post("/api/contact")
async def contact_endpoint(data: ContactRequest):
    return {"status": "success", "message": f"Thank you {data.name}, Abhisek will get back to you soon!"}