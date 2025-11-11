# Connection Error Troubleshooting Guide

## Immediate Fixes Applied

### 1. Reduced Payload Size
- ✅ Reduced context chunks from 5 to 3
- ✅ Truncated long chunks to 500 characters
- ✅ Simplified prompts to reduce size
- ✅ Added automatic payload optimization

### 2. Improved Connection Handling
- ✅ Reduced timeout from 60s to 20s
- ✅ Simplified timeout configuration
- ✅ Better session management
- ✅ Enhanced error logging

### 3. Prompt Optimization
- ✅ Reduced max prompt length to 15KB (from 32KB)
- ✅ More compact prompt format
- ✅ Automatic context truncation
- ✅ Duplicate removal

## Quick Diagnostics

### Test 1: Simple API Test
```bash
python3 quick_fix_connection.py
```

### Test 2: Full Diagnostic
```bash
python3 diagnose_api.py
```

### Test 3: Manual Test
```bash
curl -X POST https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

## Common Solutions

### Solution 1: Reduce Context Size
The app now automatically reduces context, but you can also:
- Ask more specific questions
- Process smaller documents
- Use fewer context chunks (already set to 3)

### Solution 2: Use Faster Model
Switch to fastest model:
```python
# In .env or main.py
GROQ_MODEL=llama-3.1-8b-instant
```

### Solution 3: Check Network
1. Test internet connection
2. Check firewall settings
3. Try from different network
4. Verify SSL certificates

### Solution 4: Verify API Key
1. Check at https://console.groq.com/keys
2. Verify key hasn't expired
3. Ensure key has proper permissions
4. Get new key if needed

## What Changed

### Before
- Large payloads (30-50KB)
- Long timeouts (60s)
- Complex connection logic
- 5 context chunks

### After
- Optimized payloads (<15KB)
- Shorter timeouts (20s)
- Simplified connection
- 3 context chunks
- Automatic truncation

## Monitoring

Check console/terminal for:
- Payload size warnings
- Optimization messages
- Connection retries
- Error details

## If Still Failing

1. **Run diagnostics**: `python3 diagnose_api.py`
2. **Check API status**: https://status.groq.com
3. **Verify API key**: https://console.groq.com/keys
4. **Test from command line**: Use curl test above
5. **Check network**: Test from different location

## Expected Behavior

### Successful Request
- Payload size: <15KB
- Response time: 2-5 seconds
- Status: 200 OK

### Failed Request
- Clear error message
- Retry attempts (up to 3)
- Detailed diagnostics
- User-friendly error display

## Next Steps

1. Restart the Streamlit app
2. Try with a smaller document
3. Ask a shorter question
4. Check console for diagnostics
5. Run diagnostic scripts if issues persist

The fixes should significantly reduce connection errors. If problems persist, the diagnostic tools will help identify the exact issue.

