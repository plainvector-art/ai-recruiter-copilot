import os
import logging
import numpy as np
from typing import List, Dict, Any, Union

logger = logging.getLogger("recruiter-copilot-embeddings")

class SemanticEmbeddingEngine:
    """
    Computes semantic similarity embeddings using SentenceTransformers and FAISS.
    Falls back gracefully to TF-IDF Vectorization if libraries are missing.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.use_fallback = False
        
        # Try loading SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            # Disable symlinks warning on Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"Loaded SentenceTransformer: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to TF-IDF similarity.")
            self.use_fallback = True
            
        # Initialize FAISS check
        self.faiss_available = False
        if not self.use_fallback:
            try:
                import faiss
                self.faiss_available = True
                logger.info("FAISS library loaded successfully.")
            except Exception as e:
                logger.warning(f"FAISS not available ({e}). Using numpy matrix math for vector index.")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates embedding vector for a single text.
        """
        if self.use_fallback or not self.encoder:
            # Under fallback, return a mock/placeholder vector (TF-IDF handles comparison at search time)
            return np.zeros((384,))
            
        try:
            return self.encoder.encode([text])[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros((384,))

    def compute_similarity(self, jd_text: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        Computes similarity scores between a JD and a list of candidates.
        Returns a list of float similarity scores between 0 and 1.
        """
        if not candidates:
            return []

        # Gather texts to embed
        # Holistically combine raw_text, skills, and projects for semantic comparison
        candidate_texts = []
        for c in candidates:
            # We build a descriptive context string for semantic mapping
            c_skills = ", ".join(c.get("skills", []))
            c_projects = " ".join([f"{p['title']}: {p['description']}" for p in c.get("projects", [])])
            c_experience = " ".join([f"{e['title']} at {e['company']}: {e['description']}" for e in c.get("experience", [])])
            
            context = f"{c.get('name', 'Candidate')} is a Software Engineer with skills in {c_skills}. Projects: {c_projects}. Experience: {c_experience}."
            # Fallback to raw text if context is sparse
            if len(context) < 100 and c.get("raw_text"):
                context = c.get("raw_text")
            candidate_texts.append(context)

        # Fallback Mode: TF-IDF cosine similarity
        if self.use_fallback:
            return self._compute_similarity_tfidf(jd_text, candidate_texts)

        # Normal Mode: SentenceTransformers
        try:
            jd_embedding = self.encoder.encode([jd_text]).astype('float32')
            cand_embeddings = self.encoder.encode(candidate_texts).astype('float32')
            
            # Normalize for cosine similarity
            jd_norm = jd_embedding / np.linalg.norm(jd_embedding, axis=1, keepdims=True)
            cand_norm = cand_embeddings / np.linalg.norm(cand_embeddings, axis=1, keepdims=True)

            if self.faiss_available:
                import faiss
                dimension = cand_norm.shape[1]
                # IndexFlatIP uses Inner Product (equivalent to Cosine Similarity for normalized vectors)
                index = faiss.IndexFlatIP(dimension)
                index.add(cand_norm)
                # Query index with JD embedding
                scores, indices = index.search(jd_norm, len(candidates))
                
                # Map scores back to candidate index order
                results = [0.0] * len(candidates)
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        # Normalize cosine range [-1, 1] to [0, 1]
                        results[idx] = float(max(0.0, min(1.0, (score + 1) / 2)))
                return results
            else:
                # Raw numpy matrix multiplication (cosine similarity)
                scores = np.dot(cand_norm, jd_norm.T).flatten()
                return [float(max(0.0, min(1.0, (s + 1) / 2))) for s in scores]

        except Exception as e:
            logger.error(f"Error during semantic embedding calculation: {e}. Falling back to TF-IDF.")
            return self._compute_similarity_tfidf(jd_text, candidate_texts)

    def _compute_similarity_tfidf(self, jd_text: str, candidate_texts: List[str]) -> List[float]:
        """
        Graceful fallback using scikit-learn TF-IDF Vectorizer.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            all_texts = [jd_text] + candidate_texts
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Row 0 is JD, rows 1..N are candidates
            jd_vec = tfidf_matrix[0:1]
            cand_vecs = tfidf_matrix[1:]
            
            similarities = cosine_similarity(cand_vecs, jd_vec).flatten()
            return [float(max(0.0, min(1.0, s))) for s in similarities]
        except Exception as e:
            logger.error(f"TF-IDF similarity calculations failed: {e}")
            # If all fails, return basic keyword overlap ratio
            results = []
            jd_words = set(jd_text.lower().split())
            for text in candidate_texts:
                c_words = set(text.lower().split())
                overlap = len(jd_words.intersection(c_words))
                union = len(jd_words.union(c_words))
                ratio = overlap / union if union > 0 else 0.0
                results.append(ratio)
            return results
