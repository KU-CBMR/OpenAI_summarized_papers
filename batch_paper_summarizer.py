import os, json, time, math, glob, hashlib
import pdfplumber
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


USE_OPENAI = True            
OPENAI_MODEL = "gpt-4o-mini" 
OLLAMA_MODEL = "llama3.1"    

# ==== OpenAI SDK ====
if USE_OPENAI:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    import requests


CHUNK_PAGES = 4            
MAX_CHARS_PER_CHUNK = 6000 
SLEEP_BETWEEN_CALLS = 0.6  
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


MAP_PROMPT = """
# Here write you prompt

"""

REDUCE_PROMPT = """

# Here write you prompt

{chunk_json_list}
"""

@dataclass
class Chunk:
    start_page: int
    end_page: int
    text: str

def pdf_to_chunks(pdf_path: str, pages_per_chunk: int = CHUNK_PAGES, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[Chunk]:
    chunks: List[Chunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        N = len(pdf.pages)
        i = 0
        while i < N:
            s = i
            e = min(i + pages_per_chunk, N)
            pages_text = []
            for p in range(s, e):
                t = pdf.pages[p].extract_text() or ""
                pages_text.append(t)
            block = "\n".join(pages_text).strip()
          
            if len(block) > max_chars:
                subparts = math.ceil(len(block)/max_chars)
                span = len(block)//subparts
                for k in range(subparts):
                    part = block[k*span : (k+1)*span] if k<subparts-1 else block[k*span:]
                    chunks.append(Chunk(start_page=s+1, end_page=e, text=part))
            else:
                chunks.append(Chunk(start_page=s+1, end_page=e, text=block))
            i = e
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(Exception))
def llm_call(prompt: str) -> str:
    if USE_OPENAI:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            max_tokens=1200
        )
        return resp["choices"][0]["message"]["content"].strip()
    else:
       
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature":0}})
        r.raise_for_status()
        return r.json().get("response","").strip()

def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start!=-1 and end!=-1 and end>start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        start = s.find("[")
        end = s.rfind("]")
        if start!=-1 and end!=-1 and end>start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        raise

def map_extract(paper_id: str, chunk: Chunk) -> Dict[str, Any]:
    prompt = MAP_PROMPT.replace("{slice_text}", f"[DOC={paper_id}][PAGE={chunk.start_page}-{chunk.end_page}]\n{chunk.text[:MAX_CHARS_PER_CHUNK]}")
    raw = llm_call(prompt)
    time.sleep(SLEEP_BETWEEN_CALLS)
    obj = safe_json_loads(raw)
    
    if isinstance(obj, dict):
        obj.setdefault("paper_id", paper_id)
        obj.setdefault("span_pages", f"{chunk.start_page}-{chunk.end_page}")
    return obj

def reduce_merge(paper_id: str, partial_jsons: List[Dict[str,Any]]) -> Dict[str,Any]:
    prompt = REDUCE_PROMPT.replace("{chunk_json_list}", json.dumps(partial_jsons, ensure_ascii=False))
    raw = llm_call(prompt)
    obj = safe_json_loads(raw)
    if isinstance(obj, dict):
        obj.setdefault("paper_id", paper_id)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return obj

def process_pdf(pdf_path: str) -> str:
    fname = os.path.basename(pdf_path)
    paper_id = os.path.splitext(fname)[0]
    chunks = pdf_to_chunks(pdf_path)
    partials = []
    for ck in tqdm(chunks, desc=f"Map {fname}", leave=False):
        if (ck.text or "").strip()=="":
            continue
        try:
            j = map_extract(paper_id, ck)
            partials.append(j)
        except Exception as e:
            partials.append({"paper_id": paper_id, "error": f"map_failed: {str(e)}", "span_pages": f"{ck.start_page}-{ck.end_page}"})
    merged = reduce_merge(paper_id, partials)
    out_path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return out_path

def process_path(path: str):
    pdfs = []
    if os.path.isdir(path):
        pdfs = sorted(glob.glob(os.path.join(path, "*.pdf")))
    else:
        pdfs = [path]
    print(f"Found {len(pdfs)} PDF(s).")
    summary = []
    for p in pdfs:
        try:
            out = process_pdf(p)
            summary.append({"pdf": p, "json": out, "status": "ok"})
        except Exception as e:
            summary.append({"pdf": p, "json": "", "status": f"failed: {e}"})
    with open(os.path.join(OUTPUT_DIR, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done. See outputs/")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch summarize PDFs to per-paper JSON")
    ap.add_argument("path", help="PDF path")
    args = ap.parse_args()
    process_path(args.path)
