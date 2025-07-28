import os
import time
import json
import cv2
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
import onnxruntime
import os
import json
import urllib.request
from datetime import datetime
from llama_cpp import Llama

# === CONFIG ===
PDF_FOLDER = "input"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "processing_data"
OUT_DIR = "app2/output"
ONNX_PATH = "models/yolov10n_best.onnx"
DPI = 100
CONF_THRESH = 0.25
IOU_THRESH = 0.50
IMG_SIZE = (640, 480)
BATCH_SIZE = 50
TARGET_CLASSES = [7, 10, 9]  # heading, title, text
ID2LABEL = {
    0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item', 4: 'Page-footer',
    5: 'Page-header', 6: 'Picture', 7: 'heading', 8: 'Table', 9: 'Text', 10: 'Title'
}

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD MODEL ===
session = onnxruntime.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(img_path):
    img = cv2.imread(str(img_path))
    orig = img.copy()
    img_resized = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    return img_transposed[np.newaxis, ...], orig, img.shape[1], img.shape[0]

def nms(boxes, scores):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()
    elif isinstance(indices, list):
        return [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
    return []

def process_pdf(pdf_path):
    filename = Path(pdf_path).stem
    print(f"\n📄 Processing: {filename}")
    start_time = time.time()
    doc = fitz.open(pdf_path)
    pdf_image_dir = os.path.join(IMAGE_DIR, filename)
    os.makedirs(pdf_image_dir, exist_ok=True)

    # === Convert to grayscale images ===
    for i, page in enumerate(doc):
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if pix.n == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(pdf_image_dir, f"page_{i+1}.png"), bw)

    image_paths = sorted(Path(pdf_image_dir).glob("*.png"), key=lambda x: int(x.stem.split("_")[1]))
    results = {}
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_input, originals = [], []

        for path in batch_paths:
            tensor, orig, w, h = preprocess(path)
            batch_input.append(tensor)
            originals.append((path, orig, w, h))

        batch_input = np.concatenate(batch_input, axis=0)
        outputs = session.run(None, {input_name: batch_input})[0]

        for idx, (path, orig, w, h) in enumerate(originals):
            pred = outputs[idx]
            scale_w = w / IMG_SIZE[0]
            scale_h = h / IMG_SIZE[1]
            boxes, scores, classes = [], [], []

            for det in pred:
                x1, y1, x2, y2, conf, cls = det[:6]
                if conf < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(conf))
                classes.append(int(cls))

            selected = nms(boxes, scores)
            page_num = int(path.stem.split("_")[1])
            results.setdefault(page_num, {"title": [], "heading": [], "bold_text": []})
            page = doc[page_num - 1]
            pdf_w, pdf_h = page.rect.width, page.rect.height

            for j in selected:
                x, y, bw, bh = boxes[j]
                x2, y2 = x + bw, y + bh
                cls_id = classes[j]
                label = ID2LABEL[cls_id]
                rect = fitz.Rect(x * pdf_w / w, y * pdf_h / h, x2 * pdf_w / w, y2 * pdf_h / h)

                if label == "heading":
                    if y2 > h * 0.90:
                        continue
                    too_close = False
                    for other in selected:
                        if other == j:
                            continue
                        ox, oy, ow, oh = boxes[other]
                        ox2, oy2 = ox + ow, oy + oh
                        o_cls = classes[other]
                        if o_cls in [6, 8] and 0 <= y - (oy + oh) < 20 and ox < x2 and ox2 > x:
                            too_close = True
                            break
                    if too_close:
                        continue
                    results[page_num]["heading"].append({"text": "", "box": [x, y, x2, y2]})

                elif label == "Title":
                    results[page_num]["title"].append({"text": "", "box": [x, y, x2, y2]})

                elif label == "Text":
                    spans = page.get_text("dict", clip=rect).get("blocks", [])
                    for blk in spans:
                        for ln in blk.get("lines", []):
                            for span in ln.get("spans", []):
                                txt = span.get("text", "").strip()
                                font = span.get("font", "")
                                flags = span.get("flags", 0)
                                if not txt:
                                    continue
                                if "bold" in font.lower() or (flags & 2 and not any(f in font for f in ["MI", "SY", "Symbol"])):
                                    results[page_num]["bold_text"].append({
                                        "text": txt, "box": [x, y, x2, y2]
                                    })
                elif label == "List-item":
                    spans = page.get_text("dict", clip=rect).get("blocks", [])
                    for blk in spans:
                        for ln in blk.get("lines", []):
                            for span in ln.get("spans", []):
                                txt = span.get("text", "").strip()
                                font = span.get("font", "")
                                flags = span.get("flags", 0)
                                if not txt:
                                    continue
                                if "bold" in font.lower() or (flags & 2 and not any(f in font for f in ["MI", "SY", "Symbol"])):
                                    results[page_num]["bold_text"].append({
                                        "text": txt, "box": [x, y, x2, y2]
                                    })

    # === Save Annotated PDF ===
    annotated_path = os.path.join(OUT_DIR, f"{filename}_annotated_phase1.pdf")
    for pg_num, items in results.items():
        page = doc[pg_num - 1]
        pdf_w, pdf_h = page.rect.width, page.rect.height
        for kind, entries in items.items():
            color = {"title": (1, 0, 0), "heading": (0, 0, 1), "bold_text": (0, 0.5, 0)}.get(kind, (0.5, 0.5, 0.5))
            for item in entries:
                x1, y1, x2, y2 = item["box"]
                rect = fitz.Rect(x1 * pdf_w / w, y1 * pdf_h / h, x2 * pdf_w / w, y2 * pdf_h / h)
                page.draw_rect(rect, color=color, width=1)
                page.insert_textbox(rect, kind.upper(), fontsize=6, color=color)
    # === Save PDF with All Detected YOLO Classes ===
    all_class_doc = fitz.open(pdf_path)  # Reload fresh copy of original doc
    for idx, (path, orig, w, h) in enumerate(originals):
        page_num = int(path.stem.split("_")[1])
        page = all_class_doc[page_num - 1]
        pdf_w, pdf_h = page.rect.width, page.rect.height
        pred = outputs[idx]
        scale_w = w / IMG_SIZE[0]
        scale_h = h / IMG_SIZE[1]

        for det in pred:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(x2 * scale_w), int(y2 * scale_h)
            label = ID2LABEL.get(int(cls), str(cls))
            color = tuple(np.random.default_rng(int(cls)).uniform(0, 1, size=3))
            rect = fitz.Rect(x1 * pdf_w / w, y1 * pdf_h / h, x2 * pdf_w / w, y2 * pdf_h / h)
            page.draw_rect(rect, color=color, width=0.8)
            page.insert_textbox(rect, label, fontsize=6, color=color)

    all_annotated_path = os.path.join(OUTPUT_DIR, f"{filename}_all_classes.pdf")
    all_class_doc.save(all_annotated_path)
    all_class_doc.close()
    print(f"✅ All-class annotated PDF saved: {all_annotated_path}")

    doc.save(annotated_path)
    doc.close()
    print(f"✅ Annotated: {annotated_path}")
    return filename, results

# === MAIN LOOP OVER FOLDER ===
all_results = {}
for pdf_file in sorted(Path(PDF_FOLDER).glob("*.pdf")):
    name, result = process_pdf(str(pdf_file))
    all_results[name] = result

# === SAVE COMBINED JSON ===
json_output_path = os.path.join(OUT_DIR, "phase1_output.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ All results saved in: {json_output_path}")

# PHASE 2


# === Step 1: Paths ===
EXTRACTED_JSON_PATH = "app2/output/phase1_output.json"
OUTPUT_JSON_PATH = "app2/output/challenge1b_output.json"
MODEL_DIR = "models"

GGUF_MODEL_PATH = "https://drive.google.com/file/d/1N5KrdeHA7rcJBN-qnhES17ylUUdjltOm/view?usp=sharing"
MODEL_DIR = "models"

PERSONA = "tourist"
JOB_TO_BE_DONE = "trip planning"

# === Step 2: Download model ===
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at {GGUF_MODEL_PATH}. Please mount it into Docker.")
    else:
        print("✅ Model already exists:", GGUF_MODEL_PATH)

# === Step 3: Load model ===
def load_model():
    print("🤖 Loading TinyLlama model...")
    return Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048)

# === Step 4: Prompt creator ===
def prepare_prompt(page, persona=PERSONA, task=JOB_TO_BE_DONE):
    bolds = [b['text'] for b in page.get('bold_text', []) if b['text'].strip()]
    headings = [h['text'] for h in page.get('heading', []) if h['text'].strip()]
    prompt = f"""<|system|>
You are helping a {persona} who is doing the task: {task}.
<|user|>
Here is the content of a document page:

"""
    if headings:
        prompt += "Headings:\n" + "\n".join(headings) + "\n"
    if bolds:
        prompt += "Bold Text:\n" + "\n".join(bolds) + "\n"

    prompt += """
Based on this, suggest the most relevant section title and summary.
Respond strictly in this JSON format:
{
  "section_title": "string",
  "refined_text": "string",
  "importance_rank": int
}
"""
    return prompt.strip()

# === Step 5: Inference ===
def run_inference(llm, prompt):
    output = llm(prompt, max_tokens=300, stop=["}"], echo=False)
    raw_text = output["choices"][0]["text"].strip() + "}"
    return json.loads(raw_text)

# === Step 6: Run everything ===
def main():
    download_model()
    llm = load_model()

    with open(EXTRACTED_JSON_PATH, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    metadata = {
        "input_documents": list(extracted_data.keys()),
        "persona": PERSONA,
        "job_to_be_done": JOB_TO_BE_DONE,
        "processing_timestamp": datetime.now().isoformat()
    }

    extracted_sections = []
    sub_section_analysis = []

    for doc_name, pages in extracted_data.items():
        for page_num, content in pages.items():
            prompt = prepare_prompt(content)
            print(f"📝 Inferring for page {page_num} of {doc_name}...")
            try:
                result = run_inference(llm, prompt)

                extracted_sections.append({
                    "document": doc_name,
                    "page_number": int(page_num),
                    "section_title": result.get("section_title", "N/A"),
                    "importance_rank": result.get("importance_rank", 999)
                })

                sub_section_analysis.append({
                    "document": doc_name,
                    "refined_text": result.get("refined_text", ""),
                    "page_number": int(page_num)
                })

            except Exception as e:
                print(f"⚠️ Error on page {page_num} of {doc_name}: {e}")

    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("✅ All results saved in:", OUTPUT_JSON_PATH)

if __name__ == "__main__":
    main()
