import os
import re
import json
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load env variables from root .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("recruiter-copilot-utils")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text from a PDF file using pdfplumber, falling back to pypdf.
    """
    text = ""
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            text = "\n".join(pages_text)
            if text.strip():
                logger.info(f"Successfully extracted text using pdfplumber from {pdf_path}")
                return text
    except Exception as e:
        logger.warning(f"pdfplumber failed or not installed, falling back: {e}")

    # Fallback to pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        text = "\n".join(pages_text)
        if text.strip():
            logger.info(f"Successfully extracted text using pypdf from {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"pypdf failed: {e}")

    return text

class UnifiedLLMClient:
    """
    A unified LLM interface supporting OpenAI, Gemini, and a high-quality Mock fallback.
    """
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None):
        self.provider = provider.lower() if provider else "mock"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Determine provider automatically if not specified but key is found
        if not api_key and os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif not api_key and os.getenv("GEMINI_API_KEY"):
            self.provider = "gemini"
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        logger.info(f"LLM Client initialized with provider: {self.provider}")

    def call_openai(self, prompt: str, system_instruction: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",  # Recruiter reasoning requires high quality
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Falling back to mock.")
            return self._generate_mock_response(prompt)

    def call_gemini(self, prompt: str, system_instruction: str) -> str:
        try:
            # We can use standard requests or the google-genai library if installed.
            # Using HTTP request makes it robust even if library imports have issues.
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            
            contents = []
            if system_instruction:
                # In Gemini 1.5, system instructions are set in systemInstruction config, but we can also inject them in contents
                prompt = f"System Instructions: {system_instruction}\n\nUser Request:\n{prompt}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
            else:
                logger.error(f"Gemini API returned status {response.status_code}: {response.text}")
                raise Exception(f"Gemini Error: {response.text}")
        except Exception as e:
            logger.error(f"Gemini call failed: {e}. Falling back to mock.")
            return self._generate_mock_response(prompt)

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """
        Main method to generate text based on prompt and system_instruction.
        """
        if not self.api_key or self.provider == "mock":
            return self._generate_mock_response(prompt)
            
        if self.provider == "openai":
            return self.call_openai(prompt, system_instruction)
        elif self.provider == "gemini":
            return self.call_gemini(prompt, system_instruction)
        else:
            return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generates structured recruiter explanations offline.
        Uses heuristics from the prompt text (looks for candidate and JD details).
        """
        # Search for JSON or Candidate Name in prompt
        candidate_name = "Candidate"
        match = re.search(r"Name:\s*([^\n]+)", prompt)
        if match:
            candidate_name = match.group(1).strip()
            
        skills_match = re.search(r"Skills:\s*([^\n]+)", prompt)
        skills = skills_match.group(1).strip() if skills_match else "Python, APIs"
        
        role_match = re.search(r"Role:\s*([^\n]+)", prompt)
        role = role_match.group(1).strip() if role_match else "Software Engineer"
        
        # Simple rule-based mock generation
        mock_response = {
            "strengths": [
                f"Demonstrates alignment for {role} role using key technologies: {skills}.",
                "Hands-on project history showing ability to build functional prototypes.",
                "Good technical foundation with clear evidence of self-guided study."
            ],
            "weaknesses": [
                "Could benefit from more cloud architecture (AWS/GCP) deployment experience.",
                "Limited large-scale system design or microservices ownership documentation."
            ],
            "fit_summary": f"Strong candidate demonstrating core qualifications for {role}. Technical capabilities match the baseline requirements, with solid project deliverables and structured programming skills.",
            "confidence_score": 82,
            "interview_questions": [
                f"Can you walk us through a complex feature in one of your projects using {skills}?",
                "How do you handle scaling and performance bottlenecks in your backend services?"
            ]
        }
        return json.dumps(mock_response, indent=2)
