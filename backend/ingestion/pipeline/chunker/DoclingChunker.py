import os
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    AcceleratorOptions, 
    AcceleratorDevice
)
from docling.datamodel.base_models import InputFormat

# Import RecursiveChunker for safety splitting
from backend.ingestion.pipeline.chunker.RecursiveChunker import RecursiveChunker

# Fix Windows Permission
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_CACHE_SYMLINKS"] = "0"

class DoclingChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize Docling Converter with optimized settings for hierarchy detection.
        """
        # Configure Pipeline Options
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,              # DISABLED: Won't read scanned text (Fast)
            do_table_structure=False,  # DISABLED: Won't analyze table rows/cols (Fast)
            accelerator_options=AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.CUDA  # Force GPU usage (matches docling.py logic)
            )
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Safety Splitter: Used when a semantic section (e.g. a Chapter) is too large
        self.safety_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _learn_document_structure(self, doc):
        """
        1. Finds keywords (Chapter, Section, Art).
        2. Calculates the MODE (most frequent) font size for each.
        3. Assigns a Rank based on that standard size.
        """
        # Store all sizes found for each keyword
        keyword_sizes = defaultdict(list)
        
        # Regex for "Keyword + Number" (e.g., "Section 5", "Art. IV")
        pattern_signature = r'^[a-zA-Z\u00C0-\u00FF]+\.?\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)\b'
        
        for item in doc.texts:
            text = item.text.strip()
            if not text: continue
            
            match = re.match(pattern_signature, text)
            if match:
                # Extract keyword (e.g., "Section")
                first_word = text.split()[0].lower()
                root_word = re.sub(r'[^a-zà-ÿ]', '', first_word) # Clean punctuation
                
                if len(root_word) > 2:
                    # Get Size (Rounded to nearest 0.5 to group similar sizes)
                    size = abs(item.prov[0].bbox.b - item.prov[0].bbox.t) if item.prov else 0
                    if size > 0:
                        keyword_sizes[root_word].append(round(size * 2) / 2)

        # 1. Determine Valid Keywords (Must appear >= 2 times)
        valid_keywords = {k for k, v in keyword_sizes.items() if len(v) >= 2}
        
        # 2. Calculate MODE Size per Keyword
        mode_size_map = {}
        for kw in valid_keywords:
            counts = Counter(keyword_sizes[kw])
            if counts:
                # Get the size with the highest frequency
                most_common_size = counts.most_common(1)[0][0]
                mode_size_map[kw] = most_common_size
        
        # 3. Create Rank Map (Biggest Mode Size = Rank 1)
        # Sort keywords by size descending (Big = Parent, Small = Child)
        sorted_kws = sorted(mode_size_map.keys(), key=lambda k: mode_size_map[k], reverse=True)
        
        # Map: {'chapitre': 1, 'section': 2, 'art': 3}
        semantic_rank_map = {kw: i+1 for i, kw in enumerate(sorted_kws)}
        
        return valid_keywords, semantic_rank_map

    def _smart_split(self, text, valid_keywords):
        """Splits line into multiple headers if needed."""
        if not valid_keywords:
            return [{"text": text, "is_header": False}]

        kw_str = "|".join(re.escape(k) for k in valid_keywords)
        # Regex: Matches (Keyword) (Optional Dot) (Space) (Number/Roman)
        split_pattern = rf'(?i)(\b(?:{kw_str})\.?\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)\b)'
        
        parts = re.split(split_pattern, text)
        results = []
        current_chunk = ""
        is_current_header = False

        for part in parts:
            if not part: continue
            
            # Check if part matches the header pattern
            if re.match(split_pattern, part):
                if current_chunk:
                    results.append({"text": current_chunk.strip(), "is_header": is_current_header})
                current_chunk = part
                is_current_header = True
            else:
                current_chunk += part

        if current_chunk:
            results.append({"text": current_chunk.strip(), "is_header": is_current_header})

        return results

    def chunk(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file and return structured chunks with hierarchy metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Convert Document
        result = self.converter.convert(file_path)
        doc = result.document

        # 2. Learn Rules (Mode-Based)
        valid_keywords, semantic_rank_map = self._learn_document_structure(doc)
        
        chunks = []
        path_stack = [] 
        text_buffer = []
        current_path_str = "Root"
        base_metadata = metadata or {}

        for item in doc.texts:
            text = item.text.strip()
            if not text: continue
            
            segments = self._smart_split(text, valid_keywords)
            
            for i, seg in enumerate(segments):
                seg_text = seg['text']
                is_header = seg['is_header']

                # Trust Docling if it says header and we didn't split it
                if item.label == "section_header" and len(segments) == 1:
                    is_header = True

                if is_header:
                    first_word = seg_text.split()[0].lower()
                    root_word = re.sub(r'[^a-zà-ÿ]', '', first_word)
                    
                    # Determine Rank
                    base_rank = semantic_rank_map.get(root_word, 100)
                    dynamic_rank = base_rank + i  

                    # Flush Buffer
                    if text_buffer:
                        full_content = "\n".join(text_buffer)
                        chunk_meta = base_metadata.copy()
                        chunk_meta["hierarchy_path"] = current_path_str
                        # Use safety splitter to ensure no chunk is too big
                        chunks.extend(self.safety_chunker.chunk(full_content, chunk_meta))
                        text_buffer = []

                    # Update Stack
                    while path_stack:
                        stack_top = path_stack[-1]
                        stack_kw = stack_top['text'].split()[0].lower()
                        stack_root = re.sub(r'[^a-zà-ÿ]', '', stack_kw)

                        if (stack_top['rank'] >= dynamic_rank) or (stack_root == root_word):
                            path_stack.pop()
                        else:
                            break 
                    
                    path_stack.append({'rank': dynamic_rank, 'text': seg_text})
                    current_path_str = " / ".join([p['text'] for p in path_stack])
                
                else:
                    text_buffer.append(seg_text)

        # Final Flush
        if text_buffer:
            full_content = "\n".join(text_buffer)
            chunk_meta = base_metadata.copy()
            chunk_meta["hierarchy_path"] = current_path_str
            # Use safety splitter to ensure no chunk is too big
            chunks.extend(self.safety_chunker.chunk(full_content, chunk_meta))

        # Add standard chunk indices
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(chunks)

        return chunks