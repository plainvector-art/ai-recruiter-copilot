import re
import json
import logging
from typing import Dict, Any, List, Optional
from backend.utils import UnifiedLLMClient

logger = logging.getLogger("recruiter-copilot-reasoning")

class RecruiterReasoningLayer:
    """
    Generates detailed qualitative evaluations for candidate-job fits.
    Uses LLMs (OpenAI/Gemini) if configured, otherwise falls back to intelligent rule-based templates.
    """
    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm_client = llm_client or UnifiedLLMClient(provider="mock")

    def generate_reasoning(self, candidate: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates strengths, weaknesses, fit summary, confidence score, and custom interview questions.
        """
        # If API is offline/mock, generate structured response locally
        if self.llm_client.provider == "mock" or not self.llm_client.api_key:
            return self._generate_heuristic_reasoning(candidate, jd, scores)

        try:
            return self._generate_llm_reasoning(candidate, jd, scores)
        except Exception as e:
            logger.error(f"Failed to generate LLM recruiter reasoning: {e}. Using heuristic fallback.")
            return self._generate_heuristic_reasoning(candidate, jd, scores)

    def _generate_llm_reasoning(self, candidate: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        system_instruction = (
            "You are an elite talent acquisition specialist and recruiter copilot. "
            "You evaluate candidates against job requirements and output a detailed, honest recruiting analysis. "
            "You MUST respond ONLY with a valid JSON block containing: 'strengths', 'weaknesses', 'fit_summary', "
            "'confidence_score', and 'interview_questions'."
        )
        
        # Clean data for prompt efficiency
        c_clean = {
            "name": candidate.get("name"),
            "skills": candidate.get("skills"),
            "tools": candidate.get("tools"),
            "experience": [{"title": e.get("title"), "company": e.get("company"), "years": e.get("years")} for e in candidate.get("experience", [])],
            "projects": [{"title": p.get("title"), "technologies": p.get("technologies")} for p in candidate.get("projects", [])],
            "soft_signals": candidate.get("soft_signals"),
            "behavioral_signals": candidate.get("behavioral_signals")
        }
        
        jd_clean = {
            "role": jd.get("role"),
            "seniority": jd.get("seniority"),
            "required_skills": jd.get("required_skills"),
            "preferred_skills": jd.get("preferred_skills")
        }

        prompt = (
            f"Generate a professional recruiter analysis for this candidate:\n\n"
            f"Candidate Info:\n{json.dumps(c_clean, indent=2)}\n\n"
            f"Job Description Requirements:\n{json.dumps(jd_clean, indent=2)}\n\n"
            f"Computed Match Scores:\n{json.dumps(scores, indent=2)}\n\n"
            f"Please output a JSON object matching this schema:\n"
            f"{{\n"
            f"  \"strengths\": [\"Bullet point 1 detailing a strong match\", \"Bullet point 2\", ...],\n"
            f"  \"weaknesses\": [\"Bullet point 1 detailing missing skills or gaps\", \"Bullet point 2\", ...],\n"
            f"  \"fit_summary\": \"A short narrative explaining WHY the candidate was ranked this way (string)\",\n"
            f"  \"confidence_score\": 85 (number between 0 and 100 representing your recruiter trust confidence),\n"
            f"  \"interview_questions\": [\"Custom question 1 probing their specific gaps\", \"Custom question 2\", ...]\n"
            f"}}"
        )

        response = self.llm_client.generate(prompt=prompt, system_instruction=system_instruction)
        
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                # Validate schema
                for key in ["strengths", "weaknesses", "fit_summary", "confidence_score", "interview_questions"]:
                    if key not in parsed:
                        raise ValueError(f"Missing key: {key}")
                return parsed
            except Exception as e:
                logger.error(f"Error parsing LLM reasoning JSON: {e}")
                
        raise ValueError("Invalid JSON response from LLM reasoning model")

    def _generate_heuristic_reasoning(self, candidate: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Rule-based heuristic fallback that analyzes candidate skills, experience, and projects to generate 
        high-fidelity structured recruiter evaluations.
        """
        name = candidate.get("name", "Candidate")
        role = jd.get("role", "Software Engineer")
        
        # 1. Strengths Extraction
        strengths = []
        cand_skills_l = [s.lower() for s in candidate.get("skills", [])]
        req_skills_l = [s.lower() for s in jd.get("required_skills", [])]
        
        matching_req = list(set(cand_skills_l).intersection(set(req_skills_l)))
        if matching_req:
            matching_req_cap = [s.title() for s in matching_req[:3]]
            strengths.append(f"Strong match for core required technologies, including: {', '.join(matching_req_cap)}.")
            
        if scores.get("experience", 0.0) >= 70.0:
            strengths.append("Past job titles and years of experience demonstrate strong seniority alignment.")
        elif candidate.get("internships"):
            strengths.append("Possesses practical internship experience demonstrating early professional growth.")

        if candidate.get("projects") and scores.get("projects", 0.0) >= 60.0:
            p_titles = [p["title"] for p in candidate.get("projects")[:2]]
            strengths.append(f"Hands-on project work demonstrated through active development of: {', '.join(p_titles)}.")

        if candidate.get("soft_signals", {}).get("hackathons", False):
            strengths.append("Active hackathon participant, demonstrating rapid prototyping capabilities and initiative.")
        if candidate.get("soft_signals", {}).get("leadership", False):
            strengths.append("Demonstrates leadership potential and mentorship capabilities.")
            
        if not strengths:
            strengths = [
                "Technical foundations are solid with multiple projects documented.",
                "Demonstrates key software development skills."
            ]

        # 2. Weaknesses Extraction
        weaknesses = []
        missing_req = list(set(req_skills_l) - set(cand_skills_l))
        if missing_req:
            missing_cap = [s.title() for s in missing_req[:3]]
            weaknesses.append(f"Gaps in core job requirements: missing {', '.join(missing_cap)}.")
            
        pref_skills_l = [s.lower() for s in jd.get("preferred_skills", [])]
        missing_pref = list(set(pref_skills_l) - set(cand_skills_l))
        if missing_pref:
            missing_pref_cap = [s.title() for s in missing_pref[:2]]
            weaknesses.append(f"Lacks preferred/nice-to-have capabilities: {', '.join(missing_pref_cap)}.")

        if scores.get("experience", 0.0) < 50.0:
            weaknesses.append("Years of experience are below the desired seniority benchmark for this role.")

        if not candidate.get("behavioral_signals", {}).get("github_activity", False):
            weaknesses.append("No active public portfolio or GitHub links provided to verify code quality.")

        if not weaknesses:
            weaknesses = [
                "Could benefit from deeper deployment/cloud engineering experience.",
                "Verify complex database indexing or microservices scaling skills in interviews."
            ]

        # 3. Fit Summary
        overall_score = sum(scores.values()) / len(scores) if scores else 50.0
        
        if overall_score >= 80.0:
            fit_summary = f"{name} is a top-tier fit for the {role} role. They possess exceptional skill overlap, strong project execution history, and highly relevant work experience. Excellent prospect for immediate hiring."
            confidence_score = 90
        elif overall_score >= 60.0:
            fit_summary = f"{name} is a strong, competent fit for {role}. They show solid foundations and matching skills, though there are minor gaps in seniority or specific technologies. Worth interviewing."
            confidence_score = 75
        else:
            fit_summary = f"{name} is a potential mismatch. While they have transferable skills, their profile has significant gaps in core skills and experience benchmarks relative to the {role} requirements."
            confidence_score = 55

        # 4. Custom Interview Questions
        questions = []
        if missing_req:
            questions.append(f"Can you explain how you would handle building a system with {missing_req[0].title()}, despite not having it explicitly on your resume?")
        else:
            questions.append("Can you walk us through the most technically challenging backend scaling issue you have solved?")

        if candidate.get("projects"):
            questions.append(f"In your project '{candidate['projects'][0]['title']}', how did you make architectural choices regarding performance and database structure?")
        else:
            questions.append("How do you typically structure your codebase to ensure ease of testing and deployment?")

        if scores.get("experience", 0.0) < 50.0:
            questions.append("As you transition into this role, how do you plan to quickly adapt to our scale and codebase requirements?")

        return {
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "fit_summary": fit_summary,
            "confidence_score": confidence_score,
            "interview_questions": questions[:3]
        }
