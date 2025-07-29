from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeRequest(BaseModel):
    name: str
    skills: list[str]
    experience: str
    template: str

@app.post("/generate")
def generate_documents(data: ResumeRequest):
    print("Received request", data)

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

        prompt_template = PromptTemplate(
            input_variables=["name", "skills", "experience", "template"],
            template="""
            You are an expert AI resume writer.

            Generate a professional resume in valid HTML based on the user's inputs:

            - Name: {name}
            - Skills: {skills}
            - Experience: {experience}
            - Template style: {template}

            Use clean HTML tags (<div>, <h2>, <ul>, <li>, <p>) and inline CSS if needed.
            Avoid markdown or placeholder text. Make it visually clear and job-market ready.
            """
        )

        chain = prompt_template | llm

        result = chain.invoke({
            "name": data.name,
            "skills": ", ".join(data.skills),
            "experience": data.experience,
            "template": data.template
        })

        return {"resume": result.content}
    except Exception as e:
        return {"error": str(e)}
