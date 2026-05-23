import os
import json
import logging
from backend.ranking import CandidateRankingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recruiter-copilot-verify")

def main():
    logger.info("Initializing Recruiter Copilot Verification...")
    
    # 1. Paths
    jd_path = os.path.join("data", "sample_jds", "backend_engineer.txt")
    resumes_dir = os.path.join("data", "sample_resumes")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JD
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Get Resume Paths
    resume_files = [
        "candidate_1_senior_backend.txt",
        "candidate_2_junior_frontend.txt",
        "candidate_3_keyword_stuffer.txt"
    ]
    resume_paths = [os.path.join(resumes_dir, name) for name in resume_files]

    # Initialize Engine (Mock mode for offline verification)
    engine = CandidateRankingEngine(provider="mock")

    # 2. Evaluate Candidate Files
    logger.info("Evaluating Candidate Files...")
    ranked_candidates = engine.evaluate_candidates(jd_text, resume_paths)

    # 3. Print Results
    print("\n" + "="*80)
    print("RANKED CANDIDATES (FILES)")
    print("="*80)
    for c in ranked_candidates:
        print(f"Rank {c['rank']}: {c['name']}")
        print(f"  Final Score: {c['final_score']}")
        print(f"  Sub-scores: {c['scores']}")
        print(f"  Fraud Status: {'SUSPICIOUS' if c['fraud_report']['is_suspicious'] else 'CLEAN'}")
        if c['fraud_report']['is_suspicious']:
            print(f"    Reason: {c['fraud_report']['reason']}")
        print(f"  Reasoning Fit: {c['reasoning']['fit_summary']}")
        print(f"  Confidence Score: {c['reasoning']['confidence_score']}%")
        print("-" * 80)

    # 4. Evaluate CSV Batch
    csv_path = os.path.join(resumes_dir, "candidates_batch.csv")
    logger.info("Evaluating CSV Batch...")
    ranked_csv = engine.evaluate_csv_dataset(jd_text, csv_path)

    print("\n" + "="*80)
    print("RANKED CANDIDATES (CSV BATCH)")
    print("="*80)
    for c in ranked_csv:
        print(f"Rank {c['rank']}: {c['name']} ({c['email']})")
        print(f"  Final Score: {c['final_score']}")
        print(f"  Sub-scores: {c['scores']}")
        print(f"  Fit: {c['reasoning']['fit_summary']}")
        print("-" * 80)

    # 5. Export Output
    df = engine.export_to_dataframe(ranked_candidates)
    df.to_csv(os.path.join(output_dir, "ranked_candidates.csv"), index=False)
    logger.info(f"Saved ranked candidates list to {os.path.join(output_dir, 'ranked_candidates.csv')}")

    # Export Full JSON
    with open(os.path.join(output_dir, "candidate_evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(ranked_candidates, f, indent=2)
    logger.info(f"Saved full JSON evaluation reports to {os.path.join(output_dir, 'candidate_evaluations.json')}")

    logger.info("Verification Complete. All modules working successfully!")

if __name__ == "__main__":
    main()
