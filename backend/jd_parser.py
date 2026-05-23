import re
import json
import logging
import spacy
from typing import Dict, Any, List
from backend.utils import UnifiedLLMClient

logger = logging.getLogger("recruiter-copilot-jd-parser")

# Ensure spacy model is loaded safely
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logger.warning("spaCy 'en_core_web_sm' model not found. Using blank English model.")
    nlp = spacy.blank("en")

# Ensure sentence boundaries are set by adding a sentencizer if parser/sentencizer not in pipeline
if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Predefined common technology and tools database for rule-based matching
COMMON_SKILLS = {
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust", "ruby", "php", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi", "spring boot",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd", "git", "linux", "graphql", "rest api",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "spark", "hadoop", "pandas", "numpy", "pytorch",
    "tensorflow", "scikit-learn", "keras", "openai", "bert", "llm", "nlp", "microservices", "agile", "scrum"
}

SENIORITY_KEYWORDS = {
    "Senior": ["senior", "sr", "lead", "principal", "staff", "architect", "head of", "director"],
    "Junior": ["junior", "jr", "entry", "associate", "graduate", "intern"],
    "Mid-Level": ["mid", "intermediate", "experienced"]
}

BEHAVIORAL_KEYWORDS = [
    "ownership", "collaboration", "team player", "proactive", "self-starter", "adaptable", "passion",
    "curiosity", "problem solver", "innovative", "initiative", "motivated", "integrity", "empathy"
]

COMMUNICATION_KEYWORDS = [
    "communication", "written", "verbal", "present", "stakeholders", "articulate", "interpersonal", "explain"
]

LEADERSHIP_KEYWORDS = [
    "mentor", "lead", "leadership", "guide", "manage", "supervise", "coach", "ownership", "influence"
]

DOMAIN_KEYWORDS = {
    "FinTech": ["finance", "fintech", "banking", "trading", "crypto", "blockchain", "investment"],
    "HealthTech": ["health", "healthcare", "medical", "biotech", "pharma", "clinical"],
    "E-commerce": ["retail", "e-commerce", "ecommerce", "sales", "shopify", "payment"],
    "SaaS": ["saas", "software as a service", "b2b", "subscription"],
    "AI/ML": ["artificial intelligence", "machine learning", "deep learning", "nlp", "computer vision", "llm"]
}

class JobDescriptionParser:
    """
    Parses Job Descriptions using NLP heuristics and LLM enrichment to extract key attributes.
    """
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient(provider="mock")

    def parse(self, jd_text: str) -> Dict[str, Any]:
        """
        Parses the JD text. Uses LLM if API keys are available, otherwise uses spaCy & rule-based heuristics.
        """
        # Clean text
        jd_text_clean = jd_text.strip()
        if not jd_text_clean:
            return self._empty_jd_schema()

        # If LLM is active (non-mock), use it for high accuracy extraction
        if self.llm_client.provider != "mock" and self.llm_client.api_key:
            try:
                return self._parse_with_llm(jd_text_clean)
            except Exception as e:
                logger.error(f"LLM JD parsing failed: {e}. Falling back to Rule-Based parsing.")

        return self._parse_with_rules(jd_text_clean)

    def _empty_jd_schema(self) -> Dict[str, Any]:
        return {
            "role": "Software Engineer",
            "seniority": "Mid-Level",
            "required_skills": [],
            "preferred_skills": [],
            "domain": "General Software",
            "behavioral_traits": [],
            "communication_requirements": [],
            "leadership_expectations": []
        }

    def _parse_with_rules(self, jd_text: str) -> Dict[str, Any]:
        doc = nlp(jd_text.lower())
        tokens = [token.text for token in doc]
        
        # 1. Role Title Extraction
        # Look at the first lines of the JD
        lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
        role = "Software Engineer"
        if lines:
            # Usually the first non-empty line contains the title
            for line in lines[:3]:
                if any(k in line.lower() for k in ["engineer", "developer", "manager", "lead", "architect", "analyst", "scientist", "specialist"]):
                    role = line
                    break
            if role == "Software Engineer":
                role = lines[0] # Fallback to first line
        
        # Limit role length to prevent giant strings
        if len(role) > 60:
            role = role[:60] + "..."

        # 2. Seniority Extraction
        seniority = "Mid-Level"  # Default
        found_seniority = False
        text_lower = jd_text.lower()
        for level, keywords in SENIORITY_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    seniority = level
                    found_seniority = True
                    break
            if found_seniority:
                break

        # 3. Required & Preferred Skills Extraction
        # Extract skills mentioned in the document
        skills_found = []
        for word in doc:
            if word.text in COMMON_SKILLS and word.text not in skills_found:
                skills_found.append(word.text)
        
        # Capitalize correctly
        skills_found_cleaned = []
        for s in skills_found:
            # Map back to standard casing
            for c_skill in COMMON_SKILLS:
                if c_skill == s:
                    skills_found_cleaned.append(c_skill.title() if len(c_skill) > 2 else c_skill.upper())
                    break

        # Distinguish required vs preferred skills by looking for contexts
        required_skills = []
        preferred_skills = []
        
        # Split text into sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        
        preferred_signals = ["preferred", "nice to have", "plus", "bonus", "desired", "optional", "preferred qualifications"]
        required_signals = ["required", "must have", "essential", "minimum qualifications", "expectations", "skills required"]

        for s_cleaned in skills_found_cleaned:
            s_lower = s_cleaned.lower()
            is_preferred = False
            for sent in sentences:
                if s_lower in sent.lower():
                    # check if sentence contains preferred indicators
                    if any(sig in sent.lower() for sig in preferred_signals):
                        is_preferred = True
                        break
            if is_preferred:
                preferred_skills.append(s_cleaned)
            else:
                required_skills.append(s_cleaned)

        # Ensure we have at least some required skills
        if not required_skills and preferred_skills:
            required_skills = preferred_skills
            preferred_skills = []

        # 4. Domain Extraction
        domain = "General Technology"
        for dom, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                domain = dom
                break

        # 5. Behavioral Traits
        behavioral = []
        for kw in BEHAVIORAL_KEYWORDS:
            if kw in text_lower and kw not in behavioral:
                behavioral.append(kw.capitalize())

        # 6. Communication requirements
        communication = []
        for kw in COMMUNICATION_KEYWORDS:
            if kw in text_lower:
                # Extract sentence context
                for sent in sentences:
                    if kw in sent.lower() and len(communication) < 2:
                        communication.append(sent)
        if not communication:
            communication = ["Strong written and verbal communication skills required."]

        # 7. Leadership expectations
        leadership = []
        for kw in LEADERSHIP_KEYWORDS:
            if kw in text_lower:
                for sent in sentences:
                    if kw in sent.lower() and len(leadership) < 2:
                        leadership.append(sent)
        if not leadership:
            if seniority == "Senior":
                leadership = ["Expected to guide junior developers and drive technical architecture."]
            else:
                leadership = ["Collaborates effectively with cross-functional team members."]

        return {
            "role": role,
            "seniority": seniority,
            "required_skills": required_skills[:10],
            "preferred_skills": preferred_skills[:10],
            "domain": domain,
            "behavioral_traits": behavioral[:5],
            "communication_requirements": [c[:100] + "..." if len(c) > 100 else c for c in communication],
            "leadership_expectations": [l[:100] + "..." if len(l) > 100 else l for l in leadership]
        }

    def _parse_with_llm(self, jd_text: str) -> Dict[str, Any]:
        system_instruction = (
            "You are an expert technical recruiter parser. Your job is to extract structural attributes from a Job Description text. "
            "You MUST respond ONLY with a valid JSON block containing: 'role', 'seniority', 'required_skills', 'preferred_skills', "
            "'domain', 'behavioral_traits', 'communication_requirements', 'leadership_expectations'."
        )
        
        prompt = (
            f"Please parse the following job description and format it strictly as a JSON object matching this schema:\n"
            f"{{\n"
            f"  \"role\": \"Job Title (string)\",\n"
            f"  \"seniority\": \"Seniority level, choose from: Entry-Level, Mid-Level, Senior, Lead, Executive (string)\",\n"
            f"  \"required_skills\": [\"Skill1\", \"Skill2\", ...],\n"
            f"  \"preferred_skills\": [\"Skill1\", \"Skill2\", ...],\n"
            f"  \"domain\": \"Domain e.g., FinTech, SaaS, AI/ML (string)\",\n"
            f"  \"behavioral_traits\": [\"Trait1\", \"Trait2\", ...],\n"
            f"  \"communication_requirements\": [\"Req1\", \"Req2\", ...],\n"
            f"  \"leadership_expectations\": [\"Exp1\", \"Exp2\", ...]\n"
            f"}}\n\n"
            f"Job Description Text:\n{jd_text}"
        )

        response = self.llm_client.generate(prompt=prompt, system_instruction=system_instruction)
        
        # Extract JSON from output
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed_data = json.loads(json_match.group(0))
                # Validate schema fields
                for field in ["role", "seniority", "required_skills", "preferred_skills", "domain", "behavioral_traits", "communication_requirements", "leadership_expectations"]:
                    if field not in parsed_data:
                        # Fallback to defaults or rule-based parsing if incomplete
                        parsed_data[field] = [] if "skills" in field or "traits" in field or "requirements" in field or "expectations" in field else "N/A"
                return parsed_data
            except Exception as e:
                logger.error(f"Error parsing LLM response as JSON: {e}")
                
        raise ValueError("Invalid JSON response from LLM")
