# import os
# import re
# import json
# from collections import defaultdict, Counter
# from docling.document_converter import DocumentConverter

# # Fix Windows Permission
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["HF_HUB_CACHE_SYMLINKS"] = "0"

# # ==========================================
# #  1. UNSUPERVISED LEARNING (Mode-Based)
# # ==========================================
# def learn_document_structure(doc):
#     """
#     1. Finds keywords (Chapter, Section, Art).
#     2. Calculates the MODE (most frequent) font size for each.
#     3. Assigns a Rank based on that standard size.
#     """
#     print("🧠 Learning Document DNA (Mode-Based)...")
    
#     # Store all sizes found for each keyword
#     keyword_sizes = defaultdict(list)
    
#     # Regex for "Keyword + Number" (e.g., "Section 5", "Art. IV")
#     pattern_signature = r'^[a-zA-Z\u00C0-\u00FF]+\.?\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)\b'
    
#     for item in doc.texts:
#         text = item.text.strip()
#         if not text: continue
        
#         match = re.match(pattern_signature, text)
#         if match:
#             # Extract keyword (e.g., "Section")
#             first_word = text.split()[0].lower()
#             root_word = re.sub(r'[^a-zà-ÿ]', '', first_word) # Clean punctuation
            
#             if len(root_word) > 2:
#                 # Get Size (Rounded to nearest 0.5 to group similar sizes)
#                 size = abs(item.prov[0].bbox.b - item.prov[0].bbox.t) if item.prov else 0
#                 if size > 0:
#                     keyword_sizes[root_word].append(round(size * 2) / 2)

#     # 1. Determine Valid Keywords (Must appear >= 2 times)
#     valid_keywords = {k for k, v in keyword_sizes.items() if len(v) >= 2}
    
#     # 2. Calculate MODE Size per Keyword
#     # We find the specific size that appears MOST often for this word.
#     mode_size_map = {}
#     for kw in valid_keywords:
#         counts = Counter(keyword_sizes[kw])
#         # Get the size with the highest frequency
#         most_common_size = counts.most_common(1)[0][0]
#         mode_size_map[kw] = most_common_size
    
#     # 3. Create Rank Map (Biggest Mode Size = Rank 1)
#     # Sort keywords by size descending (Big = Parent, Small = Child)
#     sorted_kws = sorted(mode_size_map.keys(), key=lambda k: mode_size_map[k], reverse=True)
    
#     # Map: {'chapitre': 1, 'section': 2, 'art': 3}
#     semantic_rank_map = {kw: i+1 for i, kw in enumerate(sorted_kws)}
    
#     print(f"   ✅ Learned Hierarchy (By Mode Size): {semantic_rank_map}")
#     print(f"   📏 Detected Standard Sizes: {mode_size_map}")
#     return valid_keywords, semantic_rank_map

# # ==========================================
# #  2. GENERIC SMART SPLITTER
# # ==========================================
# def smart_split(text, valid_keywords):
#     """Splits line into multiple headers if needed."""
#     if not valid_keywords:
#         return [{"text": text, "is_header": False}]

#     kw_str = "|".join(re.escape(k) for k in valid_keywords)
#     # Regex: Matches (Keyword) (Optional Dot) (Space) (Number/Roman)
#     # We use a capturing group to keep the delimiter in the split list
#     split_pattern = rf'(?i)(\b(?:{kw_str})\.?\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)\b)'
    
#     parts = re.split(split_pattern, text)
#     results = []
#     current_chunk = ""
#     is_current_header = False

#     for part in parts:
#         if not part: continue
        
#         # Check if part matches the header pattern
#         if re.match(split_pattern, part):
#             if current_chunk:
#                 results.append({"text": current_chunk.strip(), "is_header": is_current_header})
#             current_chunk = part
#             is_current_header = True
#         else:
#             current_chunk += part

#     if current_chunk:
#         results.append({"text": current_chunk.strip(), "is_header": is_current_header})

#     return results

# # ==========================================
# #  3. PROCESSING PIPELINE
# # ==========================================
# def run_general_pipeline(pdf_path):
#     print(f"🚀 Processing: {pdf_path}...")
    
#     converter = DocumentConverter()
#     result = converter.convert(pdf_path)
#     doc = result.document

#     # 1. LEARN RULES (Using Mode)
#     valid_keywords, semantic_rank_map = learn_document_structure(doc)
    
#     chunks = []
#     path_stack = [] 
#     text_buffer = []
#     current_path_str = "Root"
#     current_rank = 0

#     for item in doc.texts:
#         text = item.text.strip()
#         if not text: continue
        
#         # Split text (handles "Chapter I Section 1" on one line)
#         segments = smart_split(text, valid_keywords)
        
#         for i, seg in enumerate(segments):
#             seg_text = seg['text']
#             is_header = seg['is_header']

#             # Trust Docling if it says header and we didn't split it
#             if item.label == "section_header" and len(segments) == 1:
#                 is_header = True

#             if is_header:
#                 # --- A. DETERMINE RANK ---
#                 # Extract the keyword (e.g., "Art")
#                 first_word = seg_text.split()[0].lower()
#                 root_word = re.sub(r'[^a-zà-ÿ]', '', first_word)
                
#                 # Use the LEARNED RANK (Based on Mode Size)
#                 if root_word in semantic_rank_map:
#                     base_rank = semantic_rank_map[root_word]
#                 else:
#                     # Fallback for unknown headers
#                     base_rank = 100 

#                 # Adjust for position in line (Left > Right)
#                 dynamic_rank = base_rank + i  

#                 # --- B. FLUSH BUFFER ---
#                 if text_buffer:
#                     chunks.append({
#                         "path": current_path_str,
#                         "content": "\n".join(text_buffer),
#                         # "rank": current_rank
#                     })
#                     text_buffer = []

#                 # --- C. UPDATE STACK (Hierarchy Logic) ---
#                 while path_stack:
#                     stack_top = path_stack[-1]
                    
#                     # Get stack item's keyword
#                     stack_kw = stack_top['text'].split()[0].lower()
#                     stack_root = re.sub(r'[^a-zà-ÿ]', '', stack_kw)

#                     # POP IF:
#                     # 1. Stack item is "Weaker" (Higher Rank Number)
#                     # 2. OR Stack item is same keyword (Sibling) -> "Section 1" replaces "Section 2"
#                     if (stack_top['rank'] >= dynamic_rank) or (stack_root == root_word):
#                         path_stack.pop()
#                     else:
#                         break 
                
#                 path_stack.append({'rank': dynamic_rank, 'text': seg_text})
                
#                 current_path_str = " / ".join([p['text'] for p in path_stack])
#                 current_rank = dynamic_rank
            
#             else:
#                 # Content
#                 text_buffer.append(seg_text)

#     # Final Flush
#     if text_buffer:
#         chunks.append({
#             "path": current_path_str,
#             "content": "\n".join(text_buffer),
#             # "rank": current_rank
#         })

#     with open("fixed_chunks_mode.json", "w", encoding="utf-8") as f:
#         json.dump(chunks, f, indent=2, ensure_ascii=False)
        
#     print(f"✅ Done. Mode-based Ranks used: {semantic_rank_map}")

# # Run
# run_general_pipeline("data/pdfs/Liasi - Règlement d'application - 19-06-2007 - 31-12-2024.pdf")

import json
import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.chunking import HierarchicalChunker

# Fix Windows Permission
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_CACHE_SYMLINKS"] = "0"

def run_docling_optimized(pdf_path):
    print(f"🚀 Processing with Optimized Pipeline: {pdf_path}...")

    # ==========================================
    # 1. CONFIGURE PIPELINE 
    # ==========================================
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                      
        do_table_structure=True,          
        table_structure_options={"mode": TableFormerMode.ACCURATE} 
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # ==========================================
    # 2. CONVERT DOCUMENT
    # ==========================================
    result = converter.convert(pdf_path)
    doc = result.document
    print(f"   ✅ Document converted. Pages: {len(doc.pages)}")

    # ==========================================
    # 3. EXTRACT RAW CHUNKS
    # ==========================================
    chunker = HierarchicalChunker(
        merge_list_items=True
    )
    
    raw_chunks = []
    
    print("   🧠 Generating raw chunks...")
    for chunk in chunker.chunk(doc):
        # Get Path
        path_str = " / ".join(chunk.meta.headings) if chunk.meta.headings else "Root"
        
        # Get Page Number (Safe Method)
        page_number = None
        if hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                if hasattr(item, 'prov') and item.prov:
                    page_number = item.prov[0].page_no
                    break

        raw_chunks.append({
            "path": path_str,
            "content": chunk.text,
            "page": page_number
        })

    # ==========================================
    # 4. MERGE DUPLICATE PATHS (AGGREGATION)
    # ==========================================
    print(f"   🔄 Merging {len(raw_chunks)} raw chunks by path...")
    
    merged_map = {}

    for item in raw_chunks:
        path = item['path']
        content = item['content']
        page = item['page']

        if path not in merged_map:
            # Initialize new entry
            merged_map[path] = {
                "path": path,
                "content_parts": [], # List to collect text parts
                "pages": set()       # Set to collect unique pages
            }
        
        # Add data to existing entry
        if content.strip(): # Only add if there is actual text
            merged_map[path]["content_parts"].append(content)
        
        if page is not None:
            merged_map[path]["pages"].add(page)

    # ==========================================
    # 5. FORMAT FINAL OUTPUT
    # ==========================================
    final_output = []
    for path, data in merged_map.items():
        # Join content with newlines
        full_content = "\n\n".join(data["content_parts"])
        
        # Sort pages (e.g., [1, 2, 3])
        sorted_pages = sorted(list(data["pages"]))

        final_output.append({
            "path": path,
            "content": full_content,
            "pages": sorted_pages
        })

    # ==========================================
    # 6. SAVE
    # ==========================================
    output_filename = "fixed_chunks_merged.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Success! Reduced to {len(final_output)} unique sections.")
    print(f"   Saved to '{output_filename}'.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Replace with your actual file path
    pdf_file = "data/pdfs/Liasi - Règlement d'application - 19-06-2007 - 31-12-2024.pdf"
    
    if os.path.exists(pdf_file):
        run_docling_optimized(pdf_file)
    else:
        print(f"❌ Error: File not found at {pdf_file}")


