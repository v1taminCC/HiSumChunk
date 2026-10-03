#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract text from PDF files using PyPDF2 and save as txt files.
"""

# ==================== Configurable Parameters ====================
# Input PDF file path (relative to script directory, or absolute)
PDF_FILE = "pdf/example_annual_report_2023.pdf"

# Output txt file directory (relative to script directory)
TXT_DIR = "txt"

# File encoding
ENCODING = "utf-8"

# Whether to add page number markers before each page's text
ADD_PAGE_MARKER = 1

# Page marker format ({page_num} will be replaced with the page number)
PAGE_MARKER_FORMAT = ">>>PAGE {page_num}<<<"

# txt filename suffix (between filename and extension)
TXT_SUFFIX = "_ToTxt"

# Whether to show processing progress
SHOW_PROGRESS = True
# ================================================================

import os
import sys
from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: PDF file path
        
    Returns:
        Extracted text content
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        # Iterate all pages to extract text
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Add page marker based on config
                    if ADD_PAGE_MARKER:
                        text += PAGE_MARKER_FORMAT.format(page_num=page_num)
                    text += page_text
                    text += "\n"
            except Exception as e:
                if SHOW_PROGRESS:
                    print(f"Warning: Error extracting page {page_num}: {e}")
                continue
        
        return text
    except Exception as e:
        if SHOW_PROGRESS:
            print(f"Error: Failed to read PDF {pdf_path}: {e}")
        return None

def process_pdf_file(pdf_file, txt_dir):
    """
    Process a single PDF file, extract text and save to txt directory.
    
    Args:
        pdf_file: PDF file path
        txt_dir: Output txt file directory
    """
    # Ensure output directory exists
    Path(txt_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if PDF file exists
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    if not pdf_path.is_file():
        print(f"Error: Path is not a file: {pdf_path}")
        return
    
    if SHOW_PROGRESS:
        print(f"Processing: {pdf_path.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if text is None:
        if SHOW_PROGRESS:
            print(f"Skipping file: {pdf_path.name}")
        return
    
    # Generate output filename (original name + suffix + .txt)
    txt_filename = pdf_path.stem + TXT_SUFFIX + ".txt"
    txt_path = Path(txt_dir) / txt_filename
    
    # Save as txt file
    try:
        with open(txt_path, 'w', encoding=ENCODING) as f:
            f.write(text)
        if SHOW_PROGRESS:
            print(f"Saved: {txt_path}")
    except Exception as e:
        if SHOW_PROGRESS:
            print(f"Error: Failed to save file {txt_path}: {e}")

def main():
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Set input PDF file and output directory (using config section parameters)
    # If PDF_FILE is absolute, use directly; otherwise relative to script directory
    if Path(PDF_FILE).is_absolute():
        pdf_file = Path(PDF_FILE)
    else:
        pdf_file = script_dir / PDF_FILE
    
    txt_dir = script_dir / TXT_DIR
    
    # Process PDF file
    process_pdf_file(pdf_file, txt_dir)
    if SHOW_PROGRESS:
        print("\nProcessing complete!")

if __name__ == "__main__":
    main()
