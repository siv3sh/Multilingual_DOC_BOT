"""
LLM Handler for Groq API integration with multilingual support.
Handles communication with Groq API and multilingual prompt construction.
"""

import os
import json
import re
import requests
from typing import List, Dict, Any, Optional
from utils import get_language_name

class GroqHandler:
    """
    Handler for Groq API integration with multilingual document QA capabilities.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq Handler.
        
        Args:
            api_key: Groq API key (default: from environment)
            model: Groq model to use (default: mixtral-8x7b-32768)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model or os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Please set GROQ_API_KEY environment variable or create a .env file. "
                "Get your API key from: https://console.groq.com/keys"
            )
    
    def _optimize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Optimize messages to reduce payload size and avoid connection issues.
        
        Args:
            messages: Original messages
            
        Returns:
            Optimized messages with truncated context if needed
        """
        optimized = messages.copy()
        total_size = sum(len(json.dumps(msg)) for msg in optimized)
        
        # If total size is too large, truncate context in user message
        if total_size > 30000:  # ~30KB limit
            print(f"⚠️ Optimizing messages (size: {total_size/1024:.1f}KB)")
            
            if len(optimized) > 1 and 'content' in optimized[-1]:
                user_content = optimized[-1]['content']
                
                # If context is present, truncate it
                if 'DOCUMENT CONTEXT:' in user_content:
                    # Split into parts
                    context_part = user_content.split('DOCUMENT CONTEXT:')[1].split('USER QUESTION:')[0] if 'USER QUESTION:' in user_content else ''
                    query_part = user_content.split('USER QUESTION:')[1] if 'USER QUESTION:' in user_content else user_content
                    
                    # Truncate context to ~15000 chars
                    if len(context_part) > 15000:
                        context_part = context_part[:15000] + "\n\n[Context truncated for performance]"
                    
                    # Reconstruct
                    optimized[-1]['content'] = f"DOCUMENT CONTEXT:\n{context_part}\n\nUSER QUESTION: {query_part}"
        
        return optimized
    
    def _construct_multilingual_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        answer_mode: str,
    ) -> tuple:
        """
        Construct an enhanced multilingual prompt for the LLM with better instructions.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        language_name = get_language_name(language)
        
        mode_instructions = {
            "concise": "Provide a focused answer in no more than three sentences. Only include the most critical facts.",
            "detailed_with_citations": (
                "Provide a comprehensive answer. Include inline source citations in brackets referencing the source filename and chunk index, "
                "for example [source: filename.txt #3]. Mention every relevant detail from the context."
            ),
            "bullet_summary": "Respond as a bullet list. Each bullet should contain one key fact with supporting detail.",
            "step_by_step": "Respond as numbered steps that lead the reader from context to conclusion. Ensure the reasoning is explicit.",
        }
        mode_instruction = mode_instructions.get(
            answer_mode,
            mode_instructions["detailed_with_citations"],
        )

        # Enhanced system prompt with more specific instructions
        system_prompt = f"""You are an expert multilingual assistant specialized in South Indian languages and document analysis. 
Your primary task is to provide accurate, comprehensive, and contextually appropriate answers based on the provided document content.

CRITICAL INSTRUCTIONS:
1. LANGUAGE REQUIREMENT: You MUST respond EXCLUSIVELY in {language_name} ({language}). Do not mix languages.
2. CONTEXT USAGE: Use ALL available context chunks provided below. Each chunk may contain different but relevant information.
3. COMPREHENSIVENESS: Provide detailed, thorough answers. Include:
   - All relevant facts, figures, names, dates, and technical details from the context
   - Multiple perspectives if the context presents different viewpoints
   - Complete information even if it spans across multiple context chunks
4. ACCURACY: Base your answer ONLY on the provided context. Do not add information not present in the context.
5. STRUCTURE: Organize your answer logically:
   - Start with a direct answer to the question
   - Provide supporting details and evidence from the context
   - Include specific examples, numbers, or quotes when available
6. CULTURAL SENSITIVITY: Maintain appropriate cultural context and linguistic nuances for {language_name}
7. COMPLETENESS: If the question has multiple parts, address all parts comprehensively
8. CLARITY: Write clearly and concisely while being thorough
9. ANSWER MODE: {mode_instruction}

Language: {language_name} ({language})
Response Language: {language_name} ONLY"""

        # Format context with metadata - limit to top 3 chunks to reduce size
        # Filter duplicates and limit chunks
        unique_chunks = []
        seen_texts = set()
        
        # Limit to top 3 chunks for smaller payload
        limited_chunks = context_chunks[:3]
        
        for chunk in limited_chunks:
            chunk_text = chunk['text'].strip()
            
            # Truncate very long chunks
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', chunk_text.lower())
            
            # Skip if exact duplicate
            if normalized in seen_texts:
                continue
            
            # Check for high similarity
            is_duplicate = False
            for seen_text in seen_texts:
                if len(normalized) > 50 and len(seen_text) > 50:
                    text_words = set(normalized.split())
                    seen_words = set(seen_text.split())
                    if len(text_words) > 0 and len(seen_words) > 0:
                        intersection = len(text_words & seen_words)
                        union = len(text_words | seen_words)
                        similarity = intersection / union if union > 0 else 0
                        if similarity > 0.85:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique_chunks.append({**chunk, 'text': chunk_text})
                seen_texts.add(normalized)
        
        # Use unique chunks only, format more compactly
        context_text = ""
        for i, chunk in enumerate(unique_chunks, 1):
            source_label = f"{chunk['filename']} #{chunk['chunk_index']}"
            context_text += f"Context {i} (source: {source_label}): {chunk['text']}\n\n"

        # Simplified, more compact prompt to reduce size
        user_prompt = f"""Context:
{context_text.strip()}

Question: {query}

Answer in {language_name} based on the context above. Follow the instructions for the selected answer mode: {mode_instruction}."""

        return system_prompt, user_prompt
    
    def _make_api_request(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        """
        Make API request to Groq with retry logic and better error handling.
        
        Args:
            messages: List of message dictionaries
            max_retries: Maximum number of retry attempts
            
        Returns:
            Optional[str]: Generated response or None if error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Validate API key format
        if not self.api_key or len(self.api_key) < 20:
            print(f"ERROR: Invalid API key format (length: {len(self.api_key) if self.api_key else 0})")
            return None
        
        # Optimize payload size to avoid connection issues
        optimized_messages = self._optimize_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": optimized_messages,
            "temperature": 0.1,
            "max_tokens": 1000,  # Reduced for faster responses and less connection time
            "top_p": 0.9,
            "stream": False
        }
        
        # Log request details
        payload_str = json.dumps(payload)
        payload_size = len(payload_str)
        print(f"API Request - Model: {self.model}, Payload: {payload_size/1024:.1f}KB, Messages: {len(optimized_messages)}")
        
        # Warn if payload is very large
        if payload_size > 30000:  # ~30KB
            print(f"⚠️ Large payload detected ({payload_size/1024:.1f}KB), this may cause connection issues")
        
        # Simplified retry logic without session complexity
        for attempt in range(max_retries):
            try:
                # Make simple direct request (no session to avoid connection pool issues)
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=15,  # Short timeout
                    stream=False
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            return content
                        else:
                            print(f"Unexpected response format: {result}")
                            return None
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing response: {e}")
                        return None
                    finally:
                        try:
                            response.close()
                        except:
                            pass
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        print(f"Rate limited. Retrying in {wait_time} seconds...")
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded after {max_retries} attempts")
                        return None
                elif response.status_code == 401:  # Unauthorized
                    error_text = response.text
                    print(f"Authentication error: Invalid API key")
                    return f"Authentication error: Invalid API key. Please check your API key."
                elif response.status_code == 400:  # Bad request
                    error_msg = response.text
                    print(f"Bad request error: {error_msg}")
                    # Try with even smaller payload
                    if "too long" in error_msg.lower() or "token" in error_msg.lower():
                        # Further reduce payload
                        if len(optimized_messages) > 1:
                            user_msg = optimized_messages[-1]['content']
                            # Keep only query, remove context
                            if 'USER QUESTION:' in user_msg or 'Question:' in user_msg:
                                query_only = user_msg.split('USER QUESTION:')[-1] if 'USER QUESTION:' in user_msg else user_msg.split('Question:')[-1]
                                optimized_messages[-1]['content'] = f"Question: {query_only}"
                                # Retry with minimal payload
                                payload['messages'] = optimized_messages
                                if attempt < max_retries - 1:
                                    continue
                    return None
                else:
                    error_text = response.text
                    print(f"Groq API error: {response.status_code} - {error_text[:100]}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 1
                        import time
                        time.sleep(wait_time)
                        continue
                    return f"API error ({response.status_code}): {error_text[:100]}"
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, BrokenPipeError) as e:
                error_type = "Connection" if isinstance(e, requests.exceptions.ConnectionError) else "Timeout" if isinstance(e, requests.exceptions.Timeout) else "Broken pipe"
                error_details = str(e)
                print(f"{error_type} error (attempt {attempt + 1}/{max_retries}): {error_details}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"Retrying in {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
                    continue
                
                return f"Connection failed: {error_type} - {error_details}. Please check your internet connection and try again."
            except OSError as e:
                if e.errno == 32:  # Broken pipe
                    print(f"Broken pipe error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        import time
                        time.sleep(wait_time)
                        continue
                    return f"Connection interrupted. Please check your internet connection and try again."
                else:
                    print(f"OS error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 1
                        import time
                        time.sleep(wait_time)
                        continue
                    return f"Network error: {str(e)}"
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    import time
                    time.sleep(wait_time)
                    continue
                return f"Request failed: {str(e)}. Please try again."
            except Exception as e:
                print(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                if attempt == max_retries - 1:
                    return f"Error: {str(e)}. Please check the console for details."
                import time
                time.sleep(1)
                continue
        
        return None
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        answer_mode: str = "detailed_with_citations",
    ) -> str:
        """
        Generate an answer using Groq API with enhanced error handling.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            
        Returns:
            str: Generated answer
        """
        try:
            if not context_chunks:
                return self._get_no_context_response(language)
            
            # Validate query
            if not query or not query.strip():
                return self._get_no_context_response(language)
            
            # Construct prompt
            system_prompt, user_prompt = self._construct_multilingual_prompt(
                query, context_chunks, language, answer_mode
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Aggressively reduce context size to avoid connection issues
            # Limit total prompt to 15000 chars (much smaller for reliability)
            total_length = len(system_prompt) + len(user_prompt)
            max_total_length = 15000  # Reduced from 32000 for better reliability
            
            if total_length > max_total_length:
                print(f"⚠️ Prompt too long ({total_length} chars), truncating to {max_total_length}...")
                
                # Truncate context more aggressively
                if "DOCUMENT CONTEXT:" in user_prompt:
                    # Extract query first (most important)
                    if "USER QUESTION:" in user_prompt:
                        query_part = user_prompt.split("USER QUESTION:")[1]
                        # Calculate available space for context
                        available_for_context = max_total_length - len(system_prompt) - len(query_part) - 500
                        
                        if available_for_context > 1000:
                            # Extract and truncate context
                            context_part = user_prompt.split("DOCUMENT CONTEXT:")[1].split("USER QUESTION:")[0]
                            if len(context_part) > available_for_context:
                                # Keep first part of context (usually most relevant)
                                context_part = context_part[:available_for_context] + "\n\n[Context truncated]"
                            
                            user_prompt = f"""DOCUMENT CONTEXT:
{context_part}

USER QUESTION: {query_part}"""
                        else:
                            # Too little space, use minimal context
                            user_prompt = f"USER QUESTION: {query_part}"
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                
                print(f"✅ Optimized prompt size: {len(system_prompt) + len(user_prompt)} chars")
            
            # Make API request with retries
            response = self._make_api_request(messages, max_retries=3)
            
            if response and response.strip():
                # Check if response is an error message
                if response.startswith("Connection failed") or response.startswith("Authentication error") or response.startswith("API error") or response.startswith("Request error") or response.startswith("Unexpected error") or response.startswith("Broken pipe") or response.startswith("OS error"):
                    # It's an error message, return it with language-specific wrapper
                    error_msg = response
                    language_error = self._get_error_response(language)
                    return f"{language_error}\n\nTechnical details: {error_msg}"
                
                # Clean up response
                cleaned_response = response.strip()
                # Remove any markdown formatting if not needed (optional)
                return cleaned_response
            else:
                return self._get_error_response(language)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_response(language)
    
    def _get_no_context_response(self, language: str) -> str:
        """Get response when no context is available."""
        responses = {
            'hi': "क्षमा कीजिए, इस प्रश्न का उत्तर देने के लिए दस्तावेज़ में पर्याप्त जानकारी नहीं मिली।",
            'ml': "ക്ഷമിക്കണം, ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ രേഖയിൽ കണ്ടെത്താൻ കഴിഞ്ഞില്ല.",
            'ta': "மன்னிக்கவும், இந்த கேள்விக்கு பதில் அளிக்க தேவையான தகவல்களை ஆவணத்தில் காண முடியவில்லை.",
            'te': "క్షమించండి, ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి అవసరమైన సమాచారం పత్రంలో కనుగొనబడలేదు.",
            'kn': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'en': "Sorry, I couldn't find sufficient information in the document to answer this question."
        }
        return responses.get(language, responses.get(language, responses['en']))
    
    def _get_error_response(self, language: str) -> str:
        """Get response when there's an error."""
        responses = {
            'hi': "क्षमा कीजिए, उत्तर बनाने में त्रुटि हुई। कृपया फिर से प्रयास करें।",
            'ml': "ക്ഷമിക്കണം, ഉത്തരം സൃഷ്ടിക്കുന്നതിൽ ഒരു പിശക് സംഭവിച്ചു. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
            'ta': "மன்னிக்கவும், பதிலை உருவாக்குவதில் பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
            'te': "క్షమించండి, సమాధానాన్ని సృష్టించడంలో లోపం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
            'kn': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'en': "Sorry, there was an error generating the response. Please try again."
        }
        return responses.get(language, responses.get(language, responses['en']))
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, please respond with 'Connection successful'."}
            ]
            
            response = self._make_api_request(test_messages)
            return response is not None
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available Groq models.
        
        Returns:
            List[str]: List of available model names
        """
        return [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-preview",
            "mixtral-8x7b-32768",
            "llama3-70b-8192", 
            "llama3-8b-8192",
            "gemma-7b-it"
        ]
    
    def set_model(self, model: str) -> bool:
        """
        Set the Groq model to use.
        
        Args:
            model: Model name
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        available_models = self.get_available_models()
        if model in available_models:
            self.model = model
            return True
        else:
            print(f"Invalid model: {model}. Available models: {available_models}")
            return False
