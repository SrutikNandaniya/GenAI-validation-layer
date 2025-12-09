📘 AI PDF Answer Validation System
ArgyleEnigma Tech Labs — Internship Assignment Submission


🚀 Overview

This project validates AI-generated answers against a financial document (PDF).
The system checks whether each answer is:

SUPPORTED (fully matches PDF)

PARTIALLY_SUPPORTED (some match, some mismatch)

NOT_SUPPORTED (no match in PDF)

It uses embeddings + similarity search to find evidence in the PDF.

📁 Project Structure
/src
    validator.py        
    qa_samples.json       

/input-pdfs
    axis_loan1.pdf        

/screenshots             
    folder structure.png
    output.png

README.md

validation_results.json   # Final output


⚙️ Tech Stack

Python 3

PyPDF2 (PDF text extraction)

SentenceTransformers (MiniLM embeddings)

FAISS (vector search)

NumPy

JSON for input/output

📦 Installation

Run the following commands:

pip install PyPDF2 sentence-transformers faiss-cpu numpy


For Windows (FAISS):

pip install faiss-cpu-windows

▶️ How to Run the Validator
Step 1: Navigate to src folder
cd src

Step 2: Run the validator script
python validator.py --pdf ../input-pdfs/axis_loan1.pdf --qa qa_samples.json --out ../validation_results.json

What this command means:

--pdf → input PDF to validate against

--qa → JSON file containing questions & AI answers

--out → file to save validation results

📤 Output Format

The output validation_results.json contains entries like:

{
  "question": "What is the sanctioned loan amount?",
  "ai_answer": "The sanctioned loan amount is Rs. 15,00,000.",
  "validation_result": "SUPPORTED",
  "confidence_score": 0.82,
  "supporting_text": "[Page X] ... Facility Amount Rupees: 1,500,000 ..."
}

📸 Screenshots Included

Inside /screenshots, the following screenshots are provided:

Project folder structure

Running validator script (CMD)

validator.py code

qa_samples.json content

Generated validation_results.json

PDF page showing loan details

These screenshots demonstrate a working application as required by the assignment.

🧠 How the System Works

The PDF is split into text chunks

Each chunk is converted into embeddings

Each AI-generated answer is compared with the PDF using:

Semantic similarity

Numeric matching

The system assigns one of three labels:

SUPPORTED

PARTIALLY_SUPPORTED

NOT_SUPPORTED

🎯 Submission Summary

All required files included

Folder structure follows assignment instructions

PDF → Q&A → Validation pipeline works end-to-end

Output JSON provided

Screenshots for proof included