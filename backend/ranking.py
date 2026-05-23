import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from backend.jd_parser import JobDescriptionParser
from backend.resume_parser import ResumeParser
from backend.embeddings import SemanticEmbeddingEngine
from backend.scorer import HybridScorer
from backend.llm_reasoning import RecruiterReasoningLayer
from backend.utils import UnifiedLLMClient

logger = logging.getLogger("recruiter-copilot-ranking")

class CandidateRankingEngine:
    """
    Orchestrates the entire candidate evaluation pipeline:
    Parses JD and Resumes, generates embeddings, scores candidates hybridly,
    runs qualitative reasoning, and ranks candidates.
    """
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None, weights: Optional[Dict[str, float]] = None):
        self.llm_client = UnifiedLLMClient(provider=provider, api_key=api_key)
        
        self.jd_parser = JobDescriptionParser(self.llm_client)
        self.resume_parser = ResumeParser(self.llm_client)
        self.embedding_engine = SemanticEmbeddingEngine()
        self.scorer = HybridScorer(weights)
        self.reasoning_layer = RecruiterReasoningLayer(self.llm_client)

    def update_weights(self, weights: Dict[str, float]):
        """
        Dynamically updates the scoring weights (e.g. for different personas).
        """
        self.scorer = HybridScorer(weights)

    def evaluate_candidates(self, jd_text: str, resume_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Orchestrator pipeline using file paths for resumes.
        """
        # 1. Parse Job Description
        logger.info("Parsing Job Description...")
        jd_data = self.jd_parser.parse(jd_text)
        jd_data["raw_text"] = jd_text  # store raw text for embedding matching
        
        # 2. Parse Resumes
        logger.info(f"Parsing {len(resume_paths)} Resumes...")
        candidates = []
        for path in resume_paths:
            try:
                candidate = self.resume_parser.parse_file(path)
                candidates.append(candidate)
            except Exception as e:
                logger.error(f"Error parsing resume file '{path}': {e}")

        return self._run_scoring_and_reasoning_pipeline(jd_data, candidates)

    def evaluate_csv_dataset(self, jd_text: str, csv_path: str) -> List[Dict[str, Any]]:
        """
        Orchestrator pipeline using a CSV dataset of candidates.
        """
        # 1. Parse Job Description
        logger.info("Parsing Job Description...")
        jd_data = self.jd_parser.parse(jd_text)
        jd_data["raw_text"] = jd_text
        
        # 2. Parse CSV candidates
        logger.info(f"Parsing CSV candidates from {csv_path}...")
        candidates = self.resume_parser.parse_csv(csv_path)

        return self._run_scoring_and_reasoning_pipeline(jd_data, candidates)

    def evaluate_raw_candidates(self, jd_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Orchestrator pipeline using pre-parsed candidates (useful for Streamlit inputs).
        """
        # 1. Parse Job Description
        logger.info("Parsing Job Description...")
        jd_data = self.jd_parser.parse(jd_text)
        jd_data["raw_text"] = jd_text
        
        return self._run_scoring_and_reasoning_pipeline(jd_data, candidates)

    def _run_scoring_and_reasoning_pipeline(self, jd_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Inner pipeline to generate embeddings, score, reason, and sort.
        """
        if not candidates:
            logger.warning("No candidates parsed successfully. Returning empty list.")
            return []

        # 3. Compute Semantic Similarities
        logger.info("Computing semantic similarity embeddings...")
        semantic_scores = self.embedding_engine.compute_similarity(jd_data["raw_text"], candidates)

        # 4. Score and Generate Qualitative Recruiter Reasoning
        logger.info("Scoring and generating recruiter explanations...")
        ranked_candidates = []
        
        for idx, candidate in enumerate(candidates):
            semantic_score = semantic_scores[idx] if idx < len(semantic_scores) else 0.0
            
            # Compute sub-scores, final score, and check for fraud
            sub_scores, final_score, fraud_report = self.scorer.score_candidate(candidate, jd_data, semantic_score)
            
            # Generate recruiter reasoning (strengths, weaknesses, etc.)
            reasoning = self.reasoning_layer.generate_reasoning(candidate, jd_data, sub_scores)
            
            # Build full candidate report
            candidate_report = {
                "name": candidate.get("name", f"Candidate {idx+1}"),
                "email": candidate.get("email", ""),
                "phone": candidate.get("phone", ""),
                "links": candidate.get("links", {}),
                "skills": candidate.get("skills", []),
                "tools": candidate.get("tools", []),
                "experience": candidate.get("experience", []),
                "projects": candidate.get("projects", []),
                "education": candidate.get("education", []),
                "certifications": candidate.get("certifications", []),
                "internships": candidate.get("internships", []),
                "soft_signals": candidate.get("soft_signals", {}),
                "behavioral_signals": candidate.get("behavioral_signals", {}),
                "scores": sub_scores,
                "final_score": final_score,
                "reasoning": reasoning,
                "fraud_report": fraud_report
            }
            ranked_candidates.append(candidate_report)

        # 5. Sort candidates by final score descending
        ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add a rank field (1-indexed)
        for rank, c in enumerate(ranked_candidates, 1):
            c["rank"] = rank

        logger.info(f"Successfully evaluated and ranked {len(ranked_candidates)} candidates.")
        return ranked_candidates

    @staticmethod
    def export_to_dataframe(ranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Utility to convert ranked candidate reports into a structured flat DataFrame.
        """
        rows = []
        for c in ranked_candidates:
            row = {
                "Rank": c["rank"],
                "Name": c["name"],
                "Email": c["email"],
                "Phone": c["phone"],
                "Final Score": c["final_score"],
                "Semantic Similarity Match": c["scores"]["semantic"],
                "Skill Alignment Score": c["scores"]["skills"],
                "Experience Relevance Score": c["scores"]["experience"],
                "Project Quality Score": c["scores"]["projects"],
                "Behavioral Signals Score": c["scores"]["behavioral"],
                "Growth Potential Score": c["scores"]["growth"],
                "Summary": c["reasoning"]["fit_summary"],
                "Confidence Score": c["reasoning"]["confidence_score"],
                "Fraud Warning": "SUSPICIOUS" if c["fraud_report"]["is_suspicious"] else "CLEAN",
                "Fraud Details": c["fraud_report"]["reason"]
            }
            rows.append(row)
        return pd.DataFrame(rows)
