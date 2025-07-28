# Challenge 1b: Section Extraction & Reasoning Solution

## 📅 Overview
This is the solution for Challenge 1b of the Adobe India Hackathon 2025. The task is to extract and rank the most relevant sections from a collection of PDFs based on a given persona and job-to-be-done. The system must reason over visual cues like headings and bold text and generate a refined, ranked summary in a single structured JSON file.

--

## 🔖 Official Challenge Guidelines

## 📚 Submission Requirements

* **GitHub Project**: Complete working code
* **Dockerfile**: Must be in the root directory and functional
* **README.md**: Clearly documents the solution, models, and usage

### 🔧 Build Command

```bash
docker run --rm -v D:/adobe/challenge_1b/input:/app2/input -v D:/adobe/challenge_1b/output:/app2/output -v D:/adobe/challenge_1b/models:/app2/models challenge1b

```

📂 Place all PDFs in the input/ folder.
📁 Output will be generated as a single file at output/output.json.

## ⚡ Critical Constraints

| Constraint      | Requirement                |
| --------------- | -------------------------- |
| Execution Time  | ≤ 60 seconds for full doc  |
| Model Size      | ≤ 1 GB                   |
| Internet Access | Not allowed                |
| Architecture    | CPU-only (amd64)           |
| RAM             | ≤ 16 GB                    |

---

## 🔎 Key Requirements

* Automatically process all PDFs from /app2/input
* Output a single output/output.json for all documents
* Must work across simple and complex layouts
* Read-only access to /input directory
* Must use open source tools and models
* Fully containerized and portable across AMD64 systems

## 📂 Directory Structure

challenge_1b/
├── input/               # Input PDFs (read-only)
├── output/              # Final output JSON file
├── detect.py            # Main processing script
├── Dockerfile           # Docker container definition
├── models/              # Quantized TinyLlama GGUF model
└── README.md            # This documentation file

## 🧬 Solution Pipeline

##  Step 1: PDF Page Rendering
* Converts each page of each PDF into grayscale images using PyMuPDF.

##  Step 2: Visual Section Detection (YOLOv10n)
* Runs YOLOv10n ONNX model to detect:

    Headings (H1/H2/H3)
    Bold inline text

* Filters out detections near images, tables, or irrelevant elements.

##  Step 3: Prompt Construction

* Prompts are built per page using:

    Detected headings and bold text
    Given persona and job-to-be-done

##  Step 4: Lightweight LLM Reasoning

* Uses TinyLlama-1.1B-Chat (Q4_K_M GGUF) via llama-cpp-python for inference.

* Each prompt produces:
    Relevant section title (if found)
    Short summary or reasoning
    Relative importance (ranked)

##  Step 5: Final Output Generation

* A single JSON file is generated containing:
* Metadata (e.g., persona, job, timestamp, input list)
* Ranked sections per PDF with page number and refined summary

##  Libraries & Models Used

* PyMuPDF – Page rendering and text region extraction
* OpenCV – Image preprocessing
* onnxruntime – Runs YOLOv10n ONNX model
* llama-cpp-python – Quantized LLM inference on CPU
* NumPy, json, tqdm, os, datetime – Utilities


## 🧪 Testing & Validation
## 🔍 Test Command

```bash
docker run --rm \
  -v $(pwd)/input:/app2/input \
  -v $(pwd)/output:/app2/output \
  --network none \
  sectionextractor:latest
```
##  Validation Checklist
 * All PDFs in /input are processed
 * A single output/output.json is created
 * Includes metadata, ranked sections, and LLM summaries
 * Processing completes within 60 seconds
 * TinyLlama model < 1GB and CPU-only
 * No network access required
 * Compatible with AMD64 Docker platforms

##  Ready to Deploy!
* This containerized solution is optimized for offline environments and CPU-only execution. Drop your test PDFs in /input, run the container, and extract rich, reasoning-based insights with output/output.json.