import argparse
import json
import re
from typing import List, Dict

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Try to import FAISS; fall back to pure NumPy if not available
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


NUM_REGEX = re.compile(r'(\d+(\.\d+)?%?)')


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    Each page is prefixed so we know where the evidence came from.
    """
    reader = PdfReader(pdf_path)
    pages: List[str] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[Page {i+1}] {text}")

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    """
    Simple word-based chunking so each chunk is embedding-friendly.
    """
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


class PDFValidator:
    def __init__(
        self,
        pdf_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        print(f"[INFO] Loading PDF from {pdf_path} ...")
        self.raw_text = load_pdf_text(pdf_path)

        print("[INFO] Chunking text ...")
        self.chunks = chunk_text(self.raw_text)

        print(f"[INFO] Loaded {len(self.chunks)} chunks.")

        print(f"[INFO] Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        print("[INFO] Encoding chunks ...")
        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Normalize for cosine similarity
        self.embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None
            print("[WARN] FAISS not installed. Falling back to NumPy similarity.")

    def _search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Return top-k most similar chunks for the query.
        """
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        if FAISS_AVAILABLE and self.index is not None:
            scores, idx = self.index.search(q_emb, k)
            idx_list = idx[0]
            score_list = scores[0]
        else:
            # Pure NumPy cosine similarity
            sims = np.dot(self.embeddings, q_emb[0])
            idx_list = sims.argsort()[::-1][:k]
            score_list = sims[idx_list]

        results: List[Dict] = []
        for i, s in zip(idx_list, score_list):
            results.append(
                {
                    "chunk": self.chunks[int(i)],
                    "score": float(s),
                }
            )
        return results

    @staticmethod
    def _extract_numbers(text: str) -> set:
        """
        Extract numeric tokens and percentages as crude key facts.
        """
        cleaned_text = text.replace(",", "")
        matches = NUM_REGEX.findall(cleaned_text)
        nums = {m[0] for m in matches}
        return nums

    def validate(self, question: str, ai_answer: str) -> Dict:
        """
        Core validation logic:
        - retrieve relevant chunks
        - compare numbers/phrases
        - output SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED
        """
        query = f"{question} {ai_answer}"
        retrieved = self._search(query, k=5)

        if not retrieved:
            return {
                "question": question,
                "ai_answer": ai_answer,
                "validation_result": "NOT_SUPPORTED",
                "confidence_score": 0.0,
                "supporting_text": "",
            }

        support_text = "\n\n---\n\n".join([r["chunk"] for r in retrieved])
        max_sim = max(r["score"] for r in retrieved)

        ans_nums = self._extract_numbers(ai_answer)
        doc_nums = set()
        for r in retrieved[:2]:  # look at top-2 chunks for numbers
            doc_nums |= self._extract_numbers(r["chunk"])

        num_match = len(ans_nums & doc_nums)
        num_miss = len(ans_nums - doc_nums)

                # Heuristic decision rules (more lenient so clear matches become SUPPORTED)
        if num_match > 0 and max_sim >= 0.45 and num_miss == 0:
            # Numbers match and similarity is decent
            label = "SUPPORTED"
        elif num_match > 0 and max_sim >= 0.30:
            # Some numbers match but similarity is lower or a few numbers differ
            label = "PARTIALLY_SUPPORTED"
        elif max_sim >= 0.50:
            # No clear numeric match but text is still pretty similar
            label = "PARTIALLY_SUPPORTED"
        else:
            label = "NOT_SUPPORTED"


        confidence = round(float(max_sim), 2)

        return {
            "question": question,
            "ai_answer": ai_answer,
            "validation_result": label,
            "confidence_score": confidence,
            "supporting_text": support_text,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GenAI financial answer validation system"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to financial document PDF",
    )
    parser.add_argument(
        "--qa",
        required=True,
        help="Path to JSON file with predefined Q&A pairs",
    )
    parser.add_argument(
        "--out",
        default="validation_results.json",
        help="Where to save validation output JSON",
    )
    args = parser.parse_args()

    with open(args.qa, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    validator = PDFValidator(args.pdf)

    results: List[Dict] = []
    for qa in qa_list:
        question = qa["question"]
        ai_answer = qa["ai_answer"]
        print(f"[INFO] Validating: {question}")
        res = validator.validate(question, ai_answer)
        results.append(res)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
