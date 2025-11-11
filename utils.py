"""
Utility functions for multilingual document processing.
Handles text extraction, language detection, and text cleaning.
"""

import os
import tempfile
import re
from collections import Counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import pdfplumber
import docx2txt

try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore

# Language code mapping for South Indian languages
LANGUAGE_MAPPING = {
    'ml': 'Malayalam',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'hi': 'Hindi',
    'tcy': 'Tulu',  # Tulu language code
    'en': 'English'
}

# Unicode ranges for South Indian scripts
UNICODE_RANGES = {
    'hi': (0x0900, 0x097F),  # Devanagari (Hindi)
    'ml': (0x0D00, 0x0D7F),  # Malayalam
    'ta': (0x0B80, 0x0BFF),  # Tamil
    'te': (0x0C00, 0x0C7F),  # Telugu
    'kn': (0x0C80, 0x0CFF),  # Kannada
}

PADDLE_LANG_ORDER = ['en', 'ta', 'te', 'ml', 'kn', 'hi']
_paddle_instances: Dict[str, Optional["PaddleOCR"]] = {}


def _get_paddle_ocr(lang: str) -> Optional["PaddleOCR"]:
    if PaddleOCR is None:
        return None
    if lang not in _paddle_instances:
        try:
            _paddle_instances[lang] = PaddleOCR(lang=lang, show_log=False)
        except Exception:
            _paddle_instances[lang] = None
    return _paddle_instances[lang]


def paddle_ocr_text(image) -> str:
    """
    Run PaddleOCR on the provided image, iterating through supported languages.
    Returns concatenated text (or empty string if OCR fails).
    """
    if PaddleOCR is None:
        return ""

    ocr_texts: List[str] = []
    np_img = np.array(image.convert("RGB"))

    for lang in PADDLE_LANG_ORDER:
        ocr_engine = _get_paddle_ocr(lang)
        if not ocr_engine:
            continue
        try:
            results = ocr_engine.ocr(np_img, cls=True)
        except Exception:
            continue

        lang_lines: List[str] = []
        for result in results:
            if isinstance(result, list):
                for entry in result:
                    if len(entry) >= 2 and entry[1][0]:
                        candidate = entry[1][0].strip()
                        if candidate:
                            lang_lines.append(candidate)

        candidate_text = "\n".join(lang_lines).strip()
        if candidate_text:
            ocr_texts.append(candidate_text)
            if len(candidate_text) > 50:
                break  # sufficient content

    if not ocr_texts:
        return ""

    ocr_texts.sort(key=len, reverse=True)
    return ocr_texts[0]

def extract_text(file) -> str:
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text content
    """
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            return extract_from_pdf(file)
        elif file_extension == 'docx':
            return extract_from_docx(file)
        elif file_extension == 'txt':
            return extract_from_txt(file)
        elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp']:
            return extract_from_image(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: pdf, docx, txt, jpg, jpeg, png, bmp, gif, tiff, tif, webp")
    except Exception as e:
        raise Exception(f"Error extracting text from {file.name}: {str(e)}")

def extract_from_pdf(file) -> str:
    """Extract text from PDF using multiple methods with fallback strategies."""
    text_content = []
    
    # Reset file pointer
    file.seek(0)
    
    # Method 1: Try pdfplumber (best for text-based PDFs)
    try:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Primary extraction method
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(page_text.strip())
                        continue
                    
                    # Alternative extraction method
                    page_text = page.extract_text_simple()
                    if page_text and page_text.strip():
                        text_content.append(page_text.strip())
                        continue
                    
                    # Try with tables extraction if no text found
                    tables = page.extract_tables()
                    if tables:
                        table_text = []
                        for table in tables:
                            for row in table:
                                if row:
                                    row_text = ' '.join([str(cell) if cell else '' for cell in row])
                                    if row_text.strip():
                                        table_text.append(row_text.strip())
                        if table_text:
                            text_content.append('\n'.join(table_text))
                            continue
                    
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
        
        if text_content:
            return '\n'.join(text_content)
        
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
    
    # Method 2: Try with different pdfplumber settings
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                try:
                    # Try with optimized extraction parameters
                    page_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=False,
                        x_density=7.25,
                        y_density=13
                    )
                    if page_text and page_text.strip():
                        text_content.append(page_text.strip())
                except Exception:
                    continue
        
        if text_content:
            return '\n'.join(text_content)
    except Exception:
        pass
    
    # Method 3: Try PyPDF2 as fallback
    try:
        file.seek(0)
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text_content = []
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(page_text.strip())
            except Exception:
                continue
        
        if text_content:
            return '\n'.join(text_content)
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
    
    # Method 4: Try pypdf (newer library)
    try:
        file.seek(0)
        import pypdf
        pdf_reader = pypdf.PdfReader(file)
        text_content = []
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(page_text.strip())
            except Exception:
                continue
        
        if text_content:
            return '\n'.join(text_content)
    except Exception as e:
        print(f"pypdf extraction failed: {e}")
    
    # Method 5: Final fallback to OCR for image-based/scanned PDFs
    try:
        file.seek(0)
        ocr_text = ocr_pdf(file)
        if ocr_text and ocr_text.strip():
            return ocr_text
    except Exception as ocr_err:
        print(f"OCR fallback failed: {ocr_err}")
    
    # If all methods failed, return empty string with warning
    raise Exception(
        "Could not extract text from PDF using any method. "
        "The PDF may be corrupted, password-protected, or contain only images. "
        "For image-based PDFs, ensure Tesseract OCR is properly installed."
    )

def ocr_pdf(file) -> str:
    """OCR fallback: render PDF pages to images and extract text via Tesseract.
    Requires poppler (for pdf2image) and Tesseract installed on system.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        from PIL import Image
    except Exception as import_err:
        raise Exception(
            "OCR dependencies missing. Please install system packages and pip deps: "
            "poppler (system), tesseract-ocr (system), and pip install pdf2image pytesseract pillow"
        ) from import_err

    # Read file bytes for conversion
    try:
        file.seek(0)
        pdf_bytes = file.read()
    except Exception:
        # Some uploaders require getbuffer
        try:
            file.seek(0)
            pdf_bytes = file.getbuffer().tobytes()  # type: ignore[attr-defined]
        except Exception as e:
            raise Exception(f"Failed to read PDF bytes for OCR: {e}")

    # Convert PDF to images
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as conv_err:
        raise Exception(
            "Failed to render PDF pages for OCR. Ensure poppler is installed and available in PATH."
        ) from conv_err

    # Configure Tesseract language set; include Indic scripts
    tesseract_langs = 'eng+hin+mal+tam+tel+kan'
    ocr_texts: List[str] = []
    for idx, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang=tesseract_langs)
            if text and text.strip():
                ocr_texts.append(text.strip())
        except Exception as ocr_page_err:
            print(f"Multilingual OCR failed on page {idx+1}, trying English: {ocr_page_err}")
            # Fallback to English-only OCR
            try:
                text = pytesseract.image_to_string(img, lang='eng')
                if text and text.strip():
                    ocr_texts.append(text.strip())
            except Exception as eng_err:
                print(f"English OCR also failed on page {idx+1}: {eng_err}")
                # Try basic OCR without language specification
                try:
                    text = pytesseract.image_to_string(img)
                    if text and text.strip():
                        ocr_texts.append(text.strip())
                except Exception:
                    continue  # Skip this page if all methods fail
    if ocr_texts:
        return '\n\n'.join(ocr_texts)

    if PaddleOCR is not None:
        paddle_results = []
        for img in images:
            try:
                paddle_text = paddle_ocr_text(img)
            except Exception as paddle_err:
                print(f"PaddleOCR failed on a page: {paddle_err}")
                continue
            if paddle_text and paddle_text.strip():
                paddle_results.append(paddle_text.strip())
        if paddle_results:
            return '\n\n'.join(paddle_results)

    return ""

def get_language_distribution(file) -> Dict[str, int]:
    """Compute language distribution in a document. For PDFs, detects per-page languages.
    For DOCX/TXT, detects language once for the whole document.

    Args:
        file: Streamlit uploaded file object

    Returns:
        Dict[str, int]: Mapping of language code to count (pages for PDF, 1 for others)
    """
    try:
        extension = file.name.split('.')[-1].lower()
        counts: Dict[str, int] = {}
        if extension == 'pdf':
            file.seek(0)
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text() or page.extract_text_simple()
                        except Exception:
                            page_text = None
                        if page_text and page_text.strip():
                            lang = detect_language(page_text)
                            # Only count if we got a valid language
                            if lang in LANGUAGE_MAPPING:
                                counts[lang] = counts.get(lang, 0) + 1
            except Exception as e:
                print(f"Language distribution PDF error: {e}")
        elif extension in ('docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp'):
            # Reuse existing extractors for full text detection
            file.seek(0)
            text = extract_text(file)
            lang = detect_language(text or '')
            # Only count if we got a valid language
            if lang in LANGUAGE_MAPPING:
                counts[lang] = counts.get(lang, 0) + 1
        else:
            # Unsupported types handled elsewhere
            pass
        return counts
    except Exception as e:
        print(f"get_language_distribution error: {e}")
        return {}

def extract_from_docx(file) -> str:
    """Extract text from DOCX using docx2txt."""
    # Reset file pointer
    file.seek(0)
    
    # Save temporary file in a writable temp directory (works on Streamlit Cloud)
    tmp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=tmp_dir) as tmp:
        temp_path = tmp.name
        tmp.write(file.getbuffer())
    
    try:
        text = docx2txt.process(temp_path)
        return text
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_from_txt(file) -> str:
    """Extract text from TXT file."""
    # Reset file pointer
    file.seek(0)
    
    # Try different encodings
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            file.seek(0)
            return file.read().decode(encoding)
        except UnicodeDecodeError:
            continue
    
    raise Exception("Could not decode text file with any supported encoding")

def preprocess_image_for_ocr(image):
    """
    Advanced image preprocessing for better OCR accuracy.
    Applies multiple enhancement techniques.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Preprocessed image
    """
    try:
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter
        import cv2
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        try:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        except Exception:
            pass  # Skip if denoising fails
        
        # Apply adaptive thresholding for better text extraction
        try:
            # Use adaptive threshold to handle varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            gray = binary
        except Exception:
            # Fallback to simple thresholding
            try:
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                pass
        
        # Enhance contrast
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(gray)
        
        # Apply PIL enhancements
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(2.0)
        
        return enhanced_image
        
    except Exception as e:
        # If advanced preprocessing fails, return original with basic enhancements
        try:
            from PIL import ImageEnhance
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            return image
        except Exception:
            return image  # Return original if all preprocessing fails

def extract_from_image(file) -> str:
    """Extract text from image files using OCR (Tesseract) with advanced preprocessing.
    Supports multilingual text extraction including Indic scripts.
    
    Args:
        file: Streamlit uploaded file object (image)
        
    Returns:
        str: Extracted text from the image
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as import_err:
        raise Exception(
            "OCR dependencies missing. Please install: pip install Pillow pytesseract opencv-python-headless\n"
            "System requirement: Install tesseract-ocr\n"
            "  - macOS: brew install tesseract tesseract-lang\n"
            "  - Ubuntu/Debian: apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin "
            "tesseract-ocr-mal tesseract-ocr-tam tesseract-ocr-tel tesseract-ocr-kan\n"
            "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        ) from import_err
    
    try:
        # Reset file pointer
        file.seek(0)
        
        # Open image using PIL
        image = Image.open(file)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'L', 'RGBA'):
            if image.mode == 'RGBA':
                # Create white background for RGBA images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
                image = background
            else:
                image = image.convert('RGB')
        
        # Store original for fallback
        original_image = image.copy()
        
        # Apply advanced preprocessing
        try:
            preprocessed_image = preprocess_image_for_ocr(image)
        except Exception as preprocess_err:
            print(f"Advanced preprocessing failed, using original: {preprocess_err}")
            preprocessed_image = image
        
        # Configure Tesseract with support for multiple Indian languages
        # Languages: English, Hindi, Malayalam, Tamil, Telugu, Kannada
        tesseract_langs = 'eng+hin+mal+tam+tel+kan'
        
        # Try different PSM modes for better accuracy
        psm_modes = [
            ('6', 'Assume uniform block of text'),
            ('3', 'Fully automatic page segmentation'),
            ('11', 'Sparse text'),
            ('12', 'Sparse text with OSD')
        ]
        
        extracted_texts = []
        
        # Try OCR with preprocessed image first
        for psm_mode, description in psm_modes:
            try:
                custom_config = f'--oem 3 --psm {psm_mode}'
                text = pytesseract.image_to_string(
                    preprocessed_image,
                    lang=tesseract_langs,
                    config=custom_config
                )
                if text and text.strip():
                    extracted_texts.append(text.strip())
                    if len(text.strip()) > 50:  # If we got substantial text, use it
                        break
            except Exception as psm_err:
                continue
        
        # If preprocessed didn't work well, try with original image
        if not extracted_texts or len(extracted_texts[0].strip()) < 20:
            for psm_mode, description in psm_modes[:2]:  # Try first 2 modes
                try:
                    custom_config = f'--oem 3 --psm {psm_mode}'
                    text = pytesseract.image_to_string(
                        original_image,
                        lang=tesseract_langs,
                        config=custom_config
                    )
                    if text and text.strip():
                        extracted_texts.append(text.strip())
                        if len(text.strip()) > 50:
                            break
                except Exception:
                    continue
        
        # Combine all extracted texts, removing duplicates
        if extracted_texts:
            # Use the longest extraction (usually most complete)
            combined_text = max(extracted_texts, key=len)
        else:
            # Final fallback: try with English only
            try:
                combined_text = pytesseract.image_to_string(
                    preprocessed_image,
                    lang='eng',
                    config='--oem 3 --psm 6'
                )
            except Exception:
                try:
                    combined_text = pytesseract.image_to_string(original_image, lang='eng')
                except Exception:
                    combined_text = ""

        # PaddleOCR fallback for complex layouts
        if PaddleOCR is not None:
            try:
                paddle_text = paddle_ocr_text(original_image)
            except Exception as paddle_error:
                print(f"PaddleOCR image fallback failed: {paddle_error}")
                paddle_text = ""
            if paddle_text and len(paddle_text) > len(combined_text or ""):
                combined_text = paddle_text
        
        # Clean up the extracted text
        if combined_text:
            # Remove excessive whitespace
            combined_text = re.sub(r'\s+', ' ', combined_text)
            # Remove common OCR artifacts
            combined_text = re.sub(r'[^\w\s\u0900-\u097F\u0D00-\u0D7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF.,!?;:\-()\[\]{}]', '', combined_text, flags=re.UNICODE)
        
        return combined_text.strip() if combined_text else ""
        
    except Exception as e:
        if "extract_from_image" in str(e) or "OCR dependencies" in str(e):
            raise
        raise Exception(f"Error processing image: {str(e)}")

def detect_language(text: str) -> str:
    """
    Detect language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        str: Language code (e.g., 'ml', 'ta', 'te', 'kn', 'tcy', 'en')
    """
    try:
        # Clean text for better detection
        cleaned_text = clean_text_for_detection(text)
        
        # Process any length of text - no minimum requirement
        if len(cleaned_text.strip()) == 0:
            return 'en'  # Default to English if text is empty
        
        # First, use character-based detection for South Indian languages
        south_indian_lang = detect_south_indian_language(cleaned_text)
        
        if south_indian_lang != 'en':
            return south_indian_lang
        
        # If no South Indian language detected, try langdetect for English
        try:
            from langdetect import detect
            detected_lang = detect(cleaned_text)
            
            # Map detected language to our supported languages
            if detected_lang in LANGUAGE_MAPPING:
                return detected_lang
        except Exception:
            pass
        
        return 'en'  # Default fallback
            
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'  # Default fallback

def detect_south_indian_language(text: str) -> str:
    """
    Enhanced detection for Indic languages using Unicode character patterns.
    Includes Hindi (Devanagari) and major South Indian scripts.
    """
    if not text or len(text.strip()) == 0:
        return 'en'
    
    # Count characters in each relevant script
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)       # Devanagari (Hindi)
    malayalam_chars = re.findall(r'[\u0D00-\u0D7F]', text)
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', text)
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    kannada_chars = re.findall(r'[\u0C80-\u0CFF]', text)
    
    char_counts = {
        'hi': len(hindi_chars),
        'ml': len(malayalam_chars),
        'ta': len(tamil_chars),
        'te': len(telugu_chars),
        'kn': len(kannada_chars)
    }
    
    # Calculate percentages to handle mixed-language texts
    total_chars = sum(char_counts.values())
    
    if total_chars == 0:
        return 'en'  # No South Indian script detected
    
    # If one language clearly dominates (>80% of characters)
    for lang, count in char_counts.items():
        if count > 0 and (count / total_chars) > 0.8:
            return lang
    
    # Otherwise, return the language with the highest count
    if max(char_counts.values()) > 0:
        max_lang = max(char_counts, key=char_counts.get)
        return max_lang
    
    return 'en'  # Default to English

def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from language code.
    
    Args:
        lang_code: Language code (e.g., 'ml', 'ta')
        
    Returns:
        str: Language name (e.g., 'Malayalam', 'Tamil')
    """
    return LANGUAGE_MAPPING.get(lang_code, 'Unknown')

def clean_text_for_detection(text: str) -> str:
    """
    Clean text specifically for language detection.
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common English words that might interfere
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in english_words]
    
    return ' '.join(filtered_words[:100])  # Use first 100 words for detection

def clean_text(text: str) -> str:
    """
    Clean extracted text for better processing with deduplication.
    
    Args:
        text: Raw extracted text
        
    Returns:
        str: Cleaned text with duplicates removed
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Remove duplicate sentences (exact matches)
    seen_sentences = set()
    unique_lines = []
    for line in lines:
        # Normalize the line for comparison (remove extra spaces, lowercase for comparison)
        normalized = re.sub(r'\s+', ' ', line.strip())
        if normalized and normalized not in seen_sentences:
            seen_sentences.add(normalized)
            unique_lines.append(line)
    
    # Further deduplication: remove near-duplicate sentences
    # Sentences that are very similar (80%+ similarity) are considered duplicates
    final_lines = []
    for line in unique_lines:
        is_duplicate = False
        normalized_line = re.sub(r'\s+', ' ', line.strip().lower())
        if len(normalized_line) < 20:  # Skip very short lines for similarity check
            final_lines.append(line)
            continue
        
        for existing_line in final_lines:
            normalized_existing = re.sub(r'\s+', ' ', existing_line.strip().lower())
            if len(normalized_existing) < 20:
                continue
            
            # Calculate simple similarity (word overlap)
            line_words = set(normalized_line.split())
            existing_words = set(normalized_existing.split())
            
            if len(line_words) > 0 and len(existing_words) > 0:
                # Jaccard similarity
                intersection = len(line_words & existing_words)
                union = len(line_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                
                # If sentences are 80%+ similar, consider them duplicates
                if similarity > 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_lines.append(line)
    
    return '\n'.join(final_lines)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Advanced text chunking with sentence-aware splitting for better retrieval.
    Handles multilingual text including Indic scripts.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return [""]
    
    if len(text) <= chunk_size:
        return [text.strip()]
    
    chunks = []
    start = 0
    text_len = len(text)
    
    # Sentence boundary markers for multiple languages
    sentence_endings = ['.', '!', '?', '।', '॥', '।', '\n\n']
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence boundary for better context preservation
        if end < text_len:
            # Look for sentence endings in reverse order
            best_break = -1
            for ending in sentence_endings:
                # Search for sentence ending
                sentence_end = text.rfind(ending, start, end)
                if sentence_end > start + (chunk_size // 3):  # Don't break too early
                    best_break = max(best_break, sentence_end + len(ending))
            
            if best_break > start:
                end = best_break
            else:
                # If no sentence boundary found, try to break at word boundary
                # Look for whitespace near the end
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + (chunk_size // 2):
                    end = space_pos + 1
        
        chunk = text[start:end].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        if end >= text_len:
            break
        
        # Calculate overlap start, ensuring we don't go backwards
        overlap_start = end - overlap
        if overlap_start <= start:
            overlap_start = start + 1
        
        # Find a good break point for overlap (prefer word boundary)
        if overlap_start > 0:
            space_pos = text.rfind(' ', start, overlap_start + overlap)
            if space_pos > start:
                start = space_pos + 1
            else:
                start = overlap_start
        else:
            start = overlap_start
        
        # Safety check to prevent infinite loops
        if start >= text_len or start <= 0:
            break
    
    # Filter out very small chunks (unless it's the only chunk)
    if len(chunks) > 1:
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    # If we ended up with no chunks, return the whole text
    if not chunks:
        chunks = [text.strip()]
    
    return chunks

def validate_text_length(text: str, min_length: int = 0) -> bool:
    """
    Validate if extracted text meets minimum length requirements.
    Now accepts any text - always returns True unless completely empty.
    
    Args:
        text: Extracted text
        min_length: Minimum required length (default 0 - accepts anything)
        
    Returns:
        bool: True if text has any content, False only if completely empty
    """
    return text is not None and len(text.strip()) >= min_length


STOPWORDS_EN = {
    "the", "and", "for", "with", "that", "this", "from", "have", "there",
    "which", "their", "about", "into", "would", "could", "should",
}


def extract_glossary_terms(
    text: str,
    language_code: str,
    max_terms: int = 5,
) -> List[Dict[str, str]]:
    """
    Generate lightweight glossary entries by extracting salient terms and snippets.

    Args:
        text: Source document text.
        language_code: Detected language code.
        max_terms: Maximum number of terms to extract.

    Returns:
        List of glossary entries containing term and contextual snippet.
    """
    if not text or not text.strip():
        return []

    token_pattern = r'[\w\u0900-\u0D7F\u0B80-\u0BFF\u0C00-\u0CFF]+'
    tokens = re.findall(token_pattern, text)
    if not tokens:
        return []

    filtered_tokens = []
    for token in tokens:
        normalized = token.strip()
        if len(normalized) < 4:
            continue
        if normalized.isdigit():
            continue
        if language_code == "en" and normalized.lower() in STOPWORDS_EN:
            continue
        filtered_tokens.append(normalized)

    if not filtered_tokens:
        return []

    freq = Counter(filtered_tokens)
    sentences = re.split(r'(?<=[.!?।！？])\s+', text)

    glossary_entries: List[Dict[str, str]] = []
    for term, _ in freq.most_common(max_terms * 3):
        term_lower = term.lower()
        snippet = next(
            (
                sentence.strip()
                for sentence in sentences
                if term_lower in sentence.lower()
            ),
            "",
        )
        if not snippet:
            continue
        glossary_entries.append(
            {
                "term": term,
                "definition": snippet,
            }
        )
        if len(glossary_entries) >= max_terms:
            break

    return glossary_entries
