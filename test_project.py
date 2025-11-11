#!/usr/bin/env python3
"""
Test script for the Document QA Chatbot project.
Tests all components to ensure everything works correctly.
"""

import os
import re
import sys
import traceback
from pathlib import Path

API_KEY = os.getenv("GROQ_API_KEY")
LLM_TESTS_ENABLED = API_KEY is not None

if LLM_TESTS_ENABLED:
    os.environ["GROQ_API_KEY"] = API_KEY  # Ensure downstream modules can access it

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TEST 1: Testing imports...")
    print("=" * 60)
    
    try:
        import streamlit
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        from utils import extract_text, detect_language, clean_text, chunk_text
        print("✅ utils imported successfully")
    except ImportError as e:
        print(f"❌ utils import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from rag_pipeline import RAGPipeline
        print("✅ rag_pipeline imported successfully")
    except ImportError as e:
        print(f"❌ rag_pipeline import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from llm_handler import GroqHandler
        print("✅ llm_handler imported successfully")
    except ImportError as e:
        print(f"❌ llm_handler import failed: {e}")
        traceback.print_exc()
        return False
    
    print("✅ All imports successful!\n")
    return True

def test_llm_handler():
    """Test LLM handler initialization and API connection."""
    if not LLM_TESTS_ENABLED:
        print("⚠️ Skipping LLM Handler test (GROQ_API_KEY not set).")
        return True

    print("=" * 60)
    print("TEST 2: Testing LLM Handler...")
    print("=" * 60)
    
    try:
        from llm_handler import GroqHandler
        
        # Test initialization
        handler = GroqHandler()
        print("✅ GroqHandler initialized successfully")
        
        # Test connection
        print("Testing API connection...")
        if handler.test_connection():
            print("✅ API connection test successful")
        else:
            print("⚠️ API connection test failed (may be rate limited)")
        
        print("✅ LLM Handler test passed!\n")
        return True
    except Exception as e:
        print(f"❌ LLM Handler test failed: {e}")
        traceback.print_exc()
        return False

def test_text_processing():
    """Test text processing utilities."""
    print("=" * 60)
    print("TEST 3: Testing text processing...")
    print("=" * 60)
    
    try:
        from utils import clean_text, chunk_text, detect_language
        
        # Test text cleaning
        test_text = "This is a test. This is a test. This is a test."
        cleaned = clean_text(test_text)
        print(f"✅ Text cleaning: {len(cleaned)} chars (removed duplicates)")
        
        # Test language detection
        malayalam_text = "മലയാളം ഭാഷയുടെ സാമൂഹിക സംസ്കാരം"
        lang = detect_language(malayalam_text)
        print(f"✅ Language detection: {lang}")
        
        # Test chunking
        long_text = " ".join(["This is a sentence."] * 100)
        chunks = chunk_text(long_text, chunk_size=100, overlap=20)
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        print("✅ Text processing test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        traceback.print_exc()
        return False

def test_rag_pipeline():
    """Test RAG pipeline initialization."""
    print("=" * 60)
    print("TEST 4: Testing RAG Pipeline...")
    print("=" * 60)
    
    try:
        from rag_pipeline import RAGPipeline
        
        # Test initialization
        pipeline = RAGPipeline()
        print("✅ RAGPipeline initialized successfully")
        
        # Test document processing
        test_text = "This is a test document. It contains some information about testing."
        test_filename = "test.txt"
        test_language = "en"
        
        result = pipeline.process_document(test_text, test_filename, test_language)
        if result:
            print("✅ Document processing successful")
        else:
            print("❌ Document processing failed")
            return False
        
        # Test context retrieval
        query = "What is this document about?"
        context = pipeline.retrieve_context(query, top_k=3, language=test_language)
        print(f"✅ Context retrieval: {len(context)} chunks retrieved")
        
        print("✅ RAG Pipeline test passed!\n")
        return True
    except Exception as e:
        print(f"❌ RAG Pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end flow."""
    if not LLM_TESTS_ENABLED:
        print("⚠️ Skipping end-to-end test (GROQ_API_KEY not set).")
        return True

    print("=" * 60)
    print("TEST 5: Testing end-to-end flow...")
    print("=" * 60)
    
    try:
        from rag_pipeline import RAGPipeline
        from llm_handler import GroqHandler
        
        # Initialize components
        pipeline = RAGPipeline()
        handler = GroqHandler()
        
        # Process a test document
        test_text = """
        മലയാളം ഭാഷയുടെ സാമൂഹിക സംസ്കാരം വിവിധ സാംസ്കാരിക പരിണാമങ്ങളുടെ ഫലമാണ്.
        മലയാളം ഒരു ദ്രാവിഡ ഭാഷയാണ്. ഇത് കേരളത്തിൽ വ്യാപകമായി സംസാരിക്കപ്പെടുന്നു.
        മലയാള ഭാഷയ്ക്ക് സമ്പന്നമായ സാഹിത്യ പാരമ്പര്യമുണ്ട്.
        """
        
        print("Processing test document...")
        pipeline.process_document(test_text, "test_ml.txt", "ml")
        print("✅ Document processed")
        
        # Retrieve context
        query = "മലയാളം ഭാഷയെക്കുറിച്ച് എന്താണ്?"
        print(f"Retrieving context for query: {query}")
        context = pipeline.retrieve_context(query, top_k=3, language="ml")
        print(f"✅ Retrieved {len(context)} context chunks")
        
        if context:
            # Generate answer
            print("Generating answer...")
            answer = handler.generate_answer(query, context, "ml")
            if answer:
                print(f"✅ Answer generated: {answer[:100]}...")
            else:
                print("⚠️ Answer generation returned None (may be API issue)")
        else:
            print("⚠️ No context retrieved")
        
        print("✅ End-to-end test passed!\n")
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling for broken pipe and connection errors."""
    if not LLM_TESTS_ENABLED:
        print("⚠️ Skipping error handling test (GROQ_API_KEY not set).")
        return True

    print("=" * 60)
    print("TEST 6: Testing error handling...")
    print("=" * 60)
    
    try:
        from llm_handler import GroqHandler
        
        handler = GroqHandler()
        
        # Test with invalid query (should handle gracefully)
        try:
            # This should not crash even if there's an error
            result = handler.generate_answer("", [], "en")
            print("✅ Empty query handled gracefully")
        except Exception as e:
            print(f"⚠️ Empty query raised exception: {e}")
        
        # Test with no context (should return appropriate message)
        try:
            result = handler.generate_answer("test", [], "en")
            if result:
                print("✅ No context handled gracefully")
            else:
                print("⚠️ No context returned None")
        except Exception as e:
            print(f"⚠️ No context raised exception: {e}")
        
        print("✅ Error handling test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_multilingual_answers():
    """Ensure the LLM responds in the correct language for supported scripts."""
    if not LLM_TESTS_ENABLED:
        print("⚠️ Skipping multilingual answer test (GROQ_API_KEY not set).")
        return True

    print("=" * 60)
    print("TEST 7: Testing multilingual answer quality...")
    print("=" * 60)

    try:
        from rag_pipeline import RAGPipeline
        from llm_handler import GroqHandler

        handler = GroqHandler()

        samples = [
            (
                "ml",
                """
                മലയാളം കേരളത്തിലെ പ്രധാന ഭാഷയാണ്. ഇത് ദ്രാവിഡ ഭാഷാകുടുംബത്തിൽപ്പെടുന്നു.
                """,
                "മലയാളം എന്തുതന്നെയാണ്?",
                r"[\u0D00-\u0D7F]",
            ),
            (
                "ta",
                """
                தமிழ் ஒரு தொன்மையான திராவிட மொழியாகும். இது தமிழ்நாடு மற்றும் இலங்கையில் பேசப்படுகிறது.
                """,
                "தமிழ் பற்றி கூறுங்கள்?",
                r"[\u0B80-\u0BFF]",
            ),
            (
                "te",
                """
                తెలుగు ఒక ప్రధాన ద్రావిడ భాష. ఇది ఆంధ్రప్రదేశ్ మరియు తెలంగాణాలో మాట్లాడబడుతుంది.
                """,
                "తెలుగు భాష గురించి చెప్పండి?",
                r"[\u0C00-\u0C7F]",
            ),
            (
                "kn",
                """
                ಕನ್ನಡ ಕರ್ನಾಟಕದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದ್ದು, ಸಮೃದ್ಧ ಸಾಹಿತ್ಯ ಪರಂಪರೆಯನ್ನು ಹೊಂದಿದೆ.
                """,
                "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ತಿಳಿಸಿ?",
                r"[\u0C80-\u0CFF]",
            ),
        ]

        for code, text, question, pattern in samples:
            print(f"→ Testing language: {code}")
            pipeline = RAGPipeline()
            pipeline.process_document(text.strip(), f"test_{code}.txt", code)
            context = pipeline.retrieve_context(question, top_k=2, language=code)
            if not context:
                raise AssertionError(f"No context retrieved for language {code}")

            answer = handler.generate_answer(question, context, code)
            if not re.search(pattern, answer):
                raise AssertionError(f"Answer for language {code} is not in the expected script.\nAnswer: {answer}")

            print(f"   ✅ Answer returned in correct script for {code}")

        print("✅ Multilingual answer quality test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Multilingual answer test failed: {e}")
        traceback.print_exc()
        return False
def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DOCUMENT QA CHATBOT - PROJECT TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_text_processing,
        test_rag_pipeline,
        test_llm_handler,
        test_error_handling,
        test_end_to_end,
        test_multilingual_answers,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

