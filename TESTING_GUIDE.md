# Testing Guide for Document QA Chatbot

## Quick Test

### 1. Test Imports
```bash
python3 -c "from llm_handler import GroqHandler; print('✅ LLM Handler OK')"
python3 -c "from rag_pipeline import RAGPipeline; print('✅ RAG Pipeline OK')"
python3 -c "from utils import extract_text; print('✅ Utils OK')"
```

### 2. Test API Connection
1. Ensure `GROQ_API_KEY` is exported in your shell:
   ```bash
   export GROQ_API_KEY=your_actual_groq_api_key
   ```
2. Run the connection probe:
   ```bash
   python3 -c "
   from llm_handler import GroqHandler
   handler = GroqHandler()
   result = handler.test_connection()
   print('✅ API Connection:', 'OK' if result else 'FAILED')
   "
   ```

### 3. Run Full Test Suite
```bash
python3 test_project.py
```

## Manual Testing Steps

### 1. Start the Application
```bash
streamlit run main.py
```

### 2. Test Document Upload
1. Upload a PDF file from Languages 2 folder
2. Verify language detection works
3. Check that document is processed successfully

### 3. Test Query
1. Ask a question in the detected language
2. Verify that answer is generated
3. Check that answer is in the correct language

### 4. Test Error Handling
1. Test with invalid API key
2. Test with empty query
3. Test with no context
4. Test with network interruption (simulate broken pipe)

## Expected Results

### ✅ Success Indicators
- Documents upload and process successfully
- Language is detected correctly
- Queries return relevant answers
- Answers are in the correct language
- No broken pipe errors
- Error messages are user-friendly

### ❌ Failure Indicators
- Broken pipe errors
- Connection timeouts
- Invalid API key errors
- Empty responses
- Language detection failures

## Troubleshooting

### Broken Pipe Error
- **Cause**: Connection interrupted
- **Fix**: Already handled with retry logic
- **Action**: Application will retry automatically

### Connection Timeout
- **Cause**: Slow network or API issues
- **Fix**: Timeout set to 60 seconds
- **Action**: Check internet connection

### Invalid API Key
- **Cause**: Wrong or expired API key
- **Fix**: Update API key in sidebar
- **Action**: Use correct API key

### No Context Found
- **Cause**: Document not processed or query not relevant
- **Fix**: Re-upload document
- **Action**: Try different query

## Test Cases

### Test Case 1: Malayalam Document
1. Upload `Malayalam.pdf` or `Malayalam.docx`
2. Verify language detected as Malayalam
3. Ask: "മലയാളം ഭാഷയെക്കുറിച്ച് എന്താണ്?"
4. Verify answer in Malayalam

### Test Case 2: Tamil Document
1. Upload `Tamil.pdf` or `Tamil.docx`
2. Verify language detected as Tamil
3. Ask: "தமிழ் மொழி பற்றி என்ன?"
4. Verify answer in Tamil

### Test Case 3: Telugu Document
1. Upload `Telugu.pdf` or `Telugu.docx`
2. Verify language detected as Telugu
3. Ask: "తెలుగు భాష గురించి ఏమిటి?"
4. Verify answer in Telugu

### Test Case 4: Kannada Document
1. Upload `Kannada.pdf` or `Kannada.docx`
2. Verify language detected as Kannada
3. Ask: "ಕನ್ನಡ ಭಾಷೆಯ ಬಗ್ಗೆ ಏನು?"
4. Verify answer in Kannada

### Test Case 5: Error Handling
1. Test with invalid API key
2. Test with empty document
3. Test with corrupted file
4. Test with network interruption

## Performance Testing

### Response Time
- Document processing: < 30 seconds
- Query response: < 10 seconds
- API call: < 5 seconds

### Memory Usage
- Document processing: < 500MB
- Query processing: < 100MB
- Total memory: < 1GB

## Security Testing

### API Key Security
- ✅ API key not exposed in logs
- ✅ API key stored securely
- ✅ API key validated before use

### Input Validation
- ✅ File type validation
- ✅ File size validation
- ✅ Query validation

## Conclusion

The application has been tested and verified to work correctly with:
- ✅ All file types (PDF, DOCX, TXT, Images)
- ✅ All languages (Malayalam, Tamil, Telugu, Kannada)
- ✅ Error handling (Broken pipe, timeouts, etc.)
- ✅ Retry logic
- ✅ User-friendly error messages

