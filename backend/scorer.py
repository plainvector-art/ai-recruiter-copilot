import re
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("recruiter-copilot-scorer")

# Default Persona Weights
PERSONA_WEIGHTS = {
    "standard": {
        "semantic": 0.35,
        "skills": 0.20,
        "experience": 0.15,
        "projects": 0.10,
        "behavioral": 0.10,
        "growth": 0.10
    },
    "tech_lead": {
        "semantic": 0.25,
        "skills": 0.30,
        "experience": 0.15,
        "projects": 0.20,
        "behavioral": 0.05,
        "growth": 0.05
    },
    "culture_growth": {
        "semantic": 0.25,
        "skills": 0.15,
        "experience": 0.10,
        "projects": 0.10,
        "behavioral": 0.20,
        "growth": 0.20
    }
}

class HybridScorer:
    """
    Computes weighted hybrid scores for candidates based on Job Description requirements.
    Contains modular scorers for Skills, Experience, Projects, Behavioral, and Growth potential,
    and supports resume fraud detection.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or PERSONA_WEIGHTS["standard"]
        # Ensure weights sum to 1.0 (with a small margin)
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 0.001:
            logger.warning(f"Weights sum to {total_w}, normalizing to 1.0.")
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_candidate(self, candidate: Dict[str, Any], jd: Dict[str, Any], semantic_score: float) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Computes detailed score breakdown and final score for a candidate.
        Returns:
            - Dict of sub-scores (0-100)
            - Final weighted score (0-100)
            - Fraud report metadata
        """
        # 1. Semantic Match (0-100)
        # Convert 0-1 similarity value to 0-100
        semantic_val = max(0.0, min(100.0, semantic_score * 100.0))

        # 2. Skill Alignment (0-100)
        skill_val = self._score_skills(candidate.get("skills", []), jd)

        # 3. Experience Relevance (0-100)
        exp_val = self._score_experience(candidate, jd)

        # 4. Project Quality (0-100)
        project_val = self._score_projects(candidate.get("projects", []), jd)

        # 5. Behavioral Signals (0-100)
        behavioral_val = self._score_behavioral(candidate.get("soft_signals", {}), candidate.get("behavioral_signals", {}))

        # 6. Growth Potential (0-100)
        growth_val = self._score_growth(candidate, jd)

        # Compute Fraud Indicator
        fraud_report = self._detect_fraud(candidate, jd)
        if fraud_report["is_suspicious"]:
            # Recruiter Copilot flags and penalizes keyword-stuffers slightly to maintain integrity
            penalty = fraud_report["fraud_penalty"]
            semantic_val = max(0.0, semantic_val - penalty)
            skill_val = max(0.0, skill_val - penalty)

        # Calculate Final Weighted Score
        final_score = (
            self.weights["semantic"] * semantic_val +
            self.weights["skills"] * skill_val +
            self.weights["experience"] * exp_val +
            self.weights["projects"] * project_val +
            self.weights["behavioral"] * behavioral_val +
            self.weights["growth"] * growth_val
        )

        sub_scores = {
            "semantic": round(semantic_val, 1),
            "skills": round(skill_val, 1),
            "experience": round(exp_val, 1),
            "projects": round(project_val, 1),
            "behavioral": round(behavioral_val, 1),
            "growth": round(growth_val, 1)
        }

        return sub_scores, round(final_score, 1), fraud_report

    def _score_skills(self, cand_skills: List[str], jd: Dict[str, Any]) -> float:
        """
        Evaluates skills alignment, weighting required skills at 70% and preferred skills at 30%.
        """
        req_jd_skills = [s.lower() for s in jd.get("required_skills", [])]
        pref_jd_skills = [s.lower() for s in jd.get("preferred_skills", [])]
        
        c_skills_lower = [s.lower() for s in cand_skills]

        if not req_jd_skills and not pref_jd_skills:
            return 100.0  # Job description does not require any specific skills, full score.

        # Required skills match
        req_score = 1.0
        if req_jd_skills:
            req_overlap = len(set(c_skills_lower).intersection(set(req_jd_skills)))
            req_score = req_overlap / len(req_jd_skills)

        # Preferred skills match
        pref_score = 1.0
        if pref_jd_skills:
            pref_overlap = len(set(c_skills_lower).intersection(set(pref_jd_skills)))
            pref_score = pref_overlap / len(pref_jd_skills)

        # Weighted calculation
        final_skill_score = (req_score * 0.70 + pref_score * 0.30) * 100.0
        return max(0.0, min(100.0, final_skill_score))

    def _score_experience(self, candidate: Dict[str, Any], jd: Dict[str, Any]) -> float:
        """
        Evaluates seniority match and title alignment.
        """
        # Sum years of experience from candidates experience block
        exp_list = candidate.get("experience", [])
        total_years = sum([job.get("years", 0) for job in exp_list])
        
        # Extrapolate from candidate internships if total years is 0
        if total_years == 0 and candidate.get("internships"):
            total_years = len(candidate.get("internships")) * 0.5

        # Check required seniority level in JD
        jd_seniority = jd.get("seniority", "Mid-Level").lower()
        
        # Target years based on seniority
        if "senior" in jd_seniority or "lead" in jd_seniority or "architect" in jd_seniority:
            target_years = 5.0
        elif "entry" in jd_seniority or "junior" in jd_seniority or "associate" in jd_seniority:
            target_years = 1.0
        else: # Mid-Level
            target_years = 3.0

        # Years of experience score (caps at 100%)
        exp_ratio = total_years / target_years if target_years > 0 else 1.0
        # If they exceed target years, give them full points
        years_score = min(1.0, exp_ratio) * 60.0 # 60 points max for years match

        # Title alignment (40 points max)
        title_score = 0.0
        jd_role = jd.get("role", "").lower()
        
        # Check if job description role words appear in candidate job titles
        for job in exp_list:
            job_title = job.get("title", "").lower()
            # Direct title match
            if jd_role in job_title or job_title in jd_role:
                title_score = 40.0
                break
            # Partial keyword overlap
            overlap = len(set(jd_role.split()).intersection(set(job_title.split())))
            if overlap > 0:
                title_score = max(title_score, 25.0)

        # Fallback for junior developers with internships matching domain
        if title_score == 0 and candidate.get("internships"):
            for intern in candidate.get("internships"):
                intern_role = intern.get("role", "").lower()
                if any(w in intern_role for w in jd_role.split()):
                    title_score = 20.0

        return min(100.0, years_score + title_score)

    def _score_projects(self, projects: List[Dict[str, Any]], jd: Dict[str, Any]) -> float:
        """
        Rates projects based on size, relevance to JD skills, and keywords showing execution complexity.
        """
        if not projects:
            return 0.0

        score = 0.0
        # 1. Project quantity (max 30 points)
        score += min(30.0, len(projects) * 15.0)

        # 2. Technology alignment (max 40 points)
        jd_skills = [s.lower() for s in jd.get("required_skills", []) + jd.get("preferred_skills", [])]
        aligned_tech_count = 0
        
        # 3. Execution Complexity indicators (max 30 points)
        complexity_keywords = ["deploy", "aws", "gcp", "docker", "kubernetes", "cloud", "api", "microservice", "scale", "performance", "database", "ci/cd", "optimized"]
        complexity_hits = 0

        for p in projects:
            p_techs = [t.lower() for t in p.get("technologies", [])]
            # Match project technologies with JD
            aligned_tech_count += len(set(p_techs).intersection(set(jd_skills)))
            
            p_desc = p.get("description", "").lower()
            # Match complexity indicators
            complexity_hits += sum(1 for kw in complexity_keywords if kw in p_desc)

        # Scale tech alignment points
        score += min(40.0, aligned_tech_count * 10.0)
        # Scale complexity points
        score += min(30.0, complexity_hits * 7.5)

        return min(100.0, score)

    def _score_behavioral(self, soft: Dict[str, Any], behavioral: Dict[str, Any]) -> float:
        """
        Evaluates soft traits and behavioral records like leadership, volunteering, hackathons, and github.
        """
        score = 0.0

        # GitHub Activity (25 points)
        if behavioral.get("github_activity", False):
            score += 25.0

        # Leadership (25 points)
        if soft.get("leadership", False):
            score += 25.0

        # Volunteering (20 points)
        if soft.get("volunteering", False):
            score += 20.0

        # Hackathons / Initiative (30 points)
        if soft.get("hackathons", False) or soft.get("initiative", False):
            score += 30.0

        # Communication indicator
        comm_rating = soft.get("communication_rating", 3) # scale of 1-5
        score += (comm_rating - 3) * 5.0 # can boost score up to +10 or penalize down to -10

        return max(0.0, min(100.0, score))

    def _score_growth(self, candidate: Dict[str, Any], jd: Dict[str, Any]) -> float:
        """
        Computes growth potential using continuous learning indicators and academic/experience trajectories.
        """
        score = 50.0 # base score for growth stability

        # Boost for certifications / learning consistency
        if candidate.get("behavioral_signals", {}).get("learning_consistency", False) or candidate.get("certifications"):
            score += 20.0

        # Boost for leadership traits (growth indicator)
        if candidate.get("soft_signals", {}).get("leadership", False):
            score += 15.0

        # Mentorship indicators (e.g. mentor/lead keywords in experience descriptions)
        raw_text = candidate.get("raw_text", "").lower()
        if "mentor" in raw_text or "coached" in raw_text or "guided" in raw_text:
            score += 15.0

        # Education degree hierarchy (Master/PhD) can indicate deeper academic growth
        education = candidate.get("education", [])
        for edu in education:
            degree = edu.get("degree", "").lower()
            if "master" in degree or "m.s." in degree or "phd" in degree or "doctor" in degree:
                score += 10.0
                break

        return min(100.0, score)

    def _detect_fraud(self, candidate: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the candidate's resume shows signs of fraud (copy-pasting JD requirements).
        Looks for exact matching sentences, high density of JD skills without context, etc.
        """
        raw_resume = candidate.get("raw_text", "")
        if not raw_resume:
            return {"is_suspicious": False, "fraud_penalty": 0.0, "reason": ""}

        # Count exact matching long strings or paragraphs from JD (longer than 6 words)
        jd_sentences = [s.strip() for s in re.split(r'[.!?\n]', jd.get("raw_text", "")) if len(s.strip().split()) > 6]
        
        matching_sentences = 0
        copied_phrases = []
        
        for sent in jd_sentences:
            if sent.lower() in raw_resume.lower():
                matching_sentences += 1
                if len(copied_phrases) < 3:
                    copied_phrases.append(sent[:50] + "...")

        # If candidate copied more than 2 full sentences directly from JD
        is_suspicious = False
        penalty = 0.0
        reason = ""
        
        if matching_sentences >= 2:
            is_suspicious = True
            penalty = 15.0 * matching_sentences
            reason = f"High sentence duplication: copied {matching_sentences} sentences verbatim from the Job Description."
        
        # Check skill stuffing: skills are present but no projects or experience descriptions mention them
        cand_skills = candidate.get("skills", [])
        skills_stuffed = []
        if len(cand_skills) > 8:
            projects_desc = " ".join([p["description"].lower() for p in candidate.get("projects", [])])
            exp_desc = " ".join([e["description"].lower() for e in candidate.get("experience", [])])
            combined_desc = projects_desc + " " + exp_desc
            
            for skill in cand_skills:
                skill_l = skill.lower()
                # If a skill is listed but never mentioned in experience details or projects
                if len(skill_l) > 3 and skill_l not in combined_desc:
                    skills_stuffed.append(skill)
                    
            # If more than 60% of listed skills are completely absent from work experience and projects description
            if len(skills_stuffed) > 4 and len(skills_stuffed) / len(cand_skills) > 0.6:
                is_suspicious = True
                penalty = max(penalty, 25.0)
                reason += f" Possible Skill Stuffing: Skills {skills_stuffed[:4]} listed but not supported by experience/project descriptions."

        return {
            "is_suspicious": is_suspicious,
            "fraud_penalty": min(50.0, penalty),
            "reason": reason,
            "copied_phrases": copied_phrases
        }
