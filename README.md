# ✨ AI PDF Answer Validation System


## 🚀 Overview

This project validates AI-generated answers against a financial loan document (PDF).

For every question–answer pair, the system determines whether the answer is:

✅ SUPPORTED — fully matches PDF

⚠️ PARTIALLY_SUPPORTED — some match, some mismatch

❌ NOT_SUPPORTED — no relevant match found

The detection uses semantic embeddings, numeric extraction, and similarity search.

## 📁 Project Structure


## ⚙️ Tech Stack
| Component                         | Purpose                            |
|----------------------------------|------------------------------------|
| Python 3                         | Core programming language          |
| PyPDF2                           | PDF text extraction                |
| SentenceTransformers (MiniLM)    | Embedding generation               |
| FAISS                            | Fast vector similarity search      |
| NumPy                            | Numerical processing               |
| JSON                             | Input/output formats               |


## 📦 Installation

Install required libraries:
```bash
pip install PyPDF2 sentence-transformers faiss-cpu numpy
```


## For Windows FAISS:
```bash
pip install faiss-cpu-windows
```

▶️ How to Run the Validator
Step 1 — Navigate to src
```bash
cd src
```

Step 2 — Execute the script
```bash
python validator.py --pdf ../input-pdfs/axis_loan1.pdf --qa qa_samples.json --out ../validation_results.json
```

## 🔍 Argument Meaning
| Argument | Meaning                                   |
|----------|-------------------------------------------|
| --pdf    | Path to source PDF                        |
| --qa     | JSON file containing questions & answers  |
| --out    | Output file where validation results save |

📤 Output Format (validation_results.json)

## Each entry looks like:
```python
{
  "question": "What is the sanctioned loan amount?",
  "ai_answer": "The sanctioned loan amount is Rs. 15,00,000.",
  "validation_result": "SUPPORTED",
  "confidence_score": 0.82,
  "supporting_text": "[Page X] ... Facility Amount Rupees: 1,500,000 ..."
}
```
## 📸 Screenshots Included

Inside /screenshots, the following proof screenshots are available:

🗂 Project folder structure

🖥 Command-line execution of validator.py

These confirm the application works end-to-end as required.

## 🧠 How the System Works (Simplified)

Extract text from the PDF

Break it into meaningful chunks

Convert chunks → embeddings (MiniLM)

Convert Q&A → embeddings

Compare semantic similarity

Perform numeric extraction & matching

Generate decision label:

- SUPPORTED

- PARTIALLY_SUPPORTED

- NOT_SUPPORTED

## 🎯 Submission Summary

✔ Complete folder structure<br>
✔ Full PDF → Q&A → Validation pipeline<br>
✔ Final output JSON included<br>
✔ Screenshots provided<br>
✔ Easy-to-run instructions documented
