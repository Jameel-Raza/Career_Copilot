import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_utils import create_vector_store, load_n_split_documents

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class ResumeRequest(BaseModel):
    name: str
    email: str
    education: str
    skills: list[str]
    projects: str
    experience: str
    template: str

# === Load and initialize RAG on startup ===
template_dir = "./templates"
chunks = load_n_split_documents(template_dir)
vectorstore = create_vector_store(chunks)
retriever = vectorstore.as_retriever()

# === Endpoint ===
@app.post("/generate")
def generate_documents(data: ResumeRequest):
    print("Received request", data)

    try:
        # 1. Retrieve relevant content from templates
        retrieved_docs = retriever.invoke(data.template)
        print(f"Retrieved documents: {retrieved_docs}")

        # Filter to get only the relevant template based on the requested style
        relevant_doc = None
        target_filename = f"{data.template.lower()}_resume.html" # Ensure template name matches filename case
        for doc in retrieved_docs:
            if os.path.basename(doc.metadata.get('source', '')) == target_filename:
                relevant_doc = doc
                break

        if relevant_doc:
            retrieved_text = relevant_doc.page_content
            print(f"Filtered to relevant document: {relevant_doc.metadata.get('source')}")
        else:
            retrieved_text = ""
            print(f"Warning: No relevant document found for template: {data.template}. Using empty context.")

        # 2. Initialize Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

        # 3. Prompt Template
        prompt_template = PromptTemplate(
            input_variables=["name", "email", "education", "skills", "projects", "experience", "template", "context"],
            template="""
            You are an expert AI resume writer. Your primary goal is to generate a professional resume strictly following the provided HTML structure and style.

            Use this HTML structure and style as the ONLY inspiration:
            {context}

            Now generate a professional resume in valid HTML based on the user's inputs:

            - Name: {name}
            - Email: {email}
            - Education: {education}
            - Skills: {skills}
            - Projects: {projects}
            - Experience: {experience}
            - Template style: {template}

            Use clean HTML tags (<div>, <h2>, <ul>, <li>, <p>) and inline CSS if needed. Ensure the generated HTML strictly adheres to the provided style. Do NOT include markdown, placeholder text, or any content outside of valid HTML. Make it visually clear and job-market ready.
            """
        )

        # 4. Run prompt
        chain = prompt_template | llm
        result = chain.invoke({
            "name": data.name,
            "email": data.email,
            "education": data.education,
            "skills": ", ".join(data.skills),
            "projects": data.projects,
            "experience": data.experience,
            "template": data.template,
            "context": retrieved_text
        })

        return {"resume": result.content}
    except Exception as e:
        return {"error": str(e)}
