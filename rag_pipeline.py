"""
RAG Pipeline for multilingual document processing.
Handles document embedding, storage in Qdrant, and retrieval.
"""

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import numpy as np
from utils import clean_text, chunk_text

class RAGPipeline:
    """
    RAG Pipeline for multilingual document processing using Qdrant vector store.
    """
    
    def __init__(self, qdrant_url: str = None, collection_name: str = "multilingual_docs"):
        """
        Initialize RAG Pipeline.
        
        Args:
            qdrant_url: Qdrant server URL (default: localhost:6333)
            collection_name: Name of the Qdrant collection
        """
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL', 'http://localhost:6333')
        self.collection_name = collection_name
        
        # Initialize multilingual embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-large')
        normalize_flag = os.getenv('EMBEDDING_NORMALIZE', 'true').lower() != 'false'
        self.normalize_embeddings = normalize_flag
        self.embedding_model_name = model_name
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Loaded embedding model: {model_name} (dim={self.embedding_dimension})")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            # Fallback to a smaller model
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dimension = 384
                print("Loaded fallback embedding model: all-MiniLM-L6-v2")
                self.embedding_model_name = 'all-MiniLM-L6-v2'
                self.normalize_embeddings = False
            except Exception as e2:
                raise Exception(f"Failed to load any embedding model: {e2}")

        # Initialize cross-encoder reranker
        reranker_model = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-large')
        self.reranker = None
        try:
            self.reranker = CrossEncoder(reranker_model, max_length=512)
            print(f"Loaded reranker model: {reranker_model}")
        except Exception as rerank_err:
            print(f"Reranker model load failed ({rerank_err}). Continuing without reranker.")
        
        # Initialize Qdrant client
        self.qdrant_client = None
        self.use_memory = False
        
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            # Test connection
            self.qdrant_client.get_collections()
            # Create collection if it doesn't exist
            self._ensure_collection_exists()
            print("Connected to Qdrant successfully")
        except Exception as e:
            print(f"Qdrant connection failed: {e}")
            print("Using in-memory storage")
            self.use_memory = True
            self.memory_storage = []  # In-memory storage for testing
            self.qdrant_client = None
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            # Fallback: try to create collection
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
            except Exception as create_error:
                print(f"Failed to create collection: {create_error}")
    
    def _format_for_embedding(self, text: str, is_query: bool) -> str:
        """Apply model-specific formatting hints."""
        model_lower = self.embedding_model_name.lower()
        if "e5" in model_lower:
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{text}"
        if "bge-m3" in model_lower or "bge-" in model_lower:
            prefix = "query: " if is_query else "doc: "
            return f"{prefix}{text}"
        return text

    def generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            formatted_texts = [self._format_for_embedding(text, is_query) for text in texts]
            embeddings = self.embedding_model.encode(formatted_texts, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)
            if isinstance(embeddings, np.ndarray):
                return embeddings.astype(float).tolist()
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def _rerank_contexts(self, query: str, contexts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking to the retrieved contexts."""
        if not self.reranker or not contexts:
            return contexts

        try:
            pairs = [(query, ctx['text']) for ctx in contexts]
            scores = self.reranker.predict(pairs)
            for ctx, score in zip(contexts, scores):
                ctx['rerank_score'] = float(score)
            contexts.sort(key=lambda item: item.get('rerank_score', item.get('score', 0.0)), reverse=True)
            return contexts[:top_k]
        except Exception as rerank_error:
            print(f"Reranking failed: {rerank_error}")
            return contexts[:top_k]
    
    def process_document(
        self,
        text: str,
        filename: str,
        language: str,
        uploaded_at: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Process and store a document in the vector store with optimized chunking.
        
        Args:
            text: Document text content
            filename: Name of the document file
            language: Detected language code
            uploaded_at: Unix timestamp when the document was ingested
            extra_metadata: Optional metadata to merge into each chunk payload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            timestamp = uploaded_at or time.time()
            extra_metadata = extra_metadata or {}
            # Clean the text
            cleaned_text = clean_text(text) if text else text
            
            # Process even if text is minimal or empty
            if not cleaned_text or len(cleaned_text.strip()) == 0:
                print("Warning: Processing document with minimal/no text")
                cleaned_text = text if text else "[Empty Document]"
            
            # Optimize chunk size based on text length and language
            # Larger chunks for longer documents, smaller for shorter ones
            text_length = len(cleaned_text)
            if text_length < 1000:
                chunk_size = 400
                overlap = 50
            elif text_length < 10000:
                chunk_size = 600
                overlap = 75
            else:
                chunk_size = 800
                overlap = 100
            
            # Chunk the text with optimized parameters
            chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
            
            # If no chunks generated, create at least one chunk with whatever we have
            if not chunks:
                print("Warning: No chunks generated, creating single chunk")
                chunks = [cleaned_text] if cleaned_text else ["[No content]"]
            
            # Filter out very small chunks that might not be useful
            chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
            
            if not chunks:
                chunks = [cleaned_text] if cleaned_text else ["[No content]"]
            
            # Remove duplicate chunks before processing
            unique_chunks = []
            seen_chunks = set()
            
            for chunk in chunks:
                # Normalize chunk for comparison
                normalized_chunk = re.sub(r'\s+', ' ', chunk.strip().lower())
                
                # Skip exact duplicates
                if normalized_chunk in seen_chunks:
                    continue
                
                # Check for high similarity with existing chunks (80%+ similarity)
                is_duplicate = False
                if len(normalized_chunk) > 50:  # Only check longer chunks
                    for seen_chunk in seen_chunks:
                        if len(seen_chunk) > 50:
                            # Word-level similarity check
                            chunk_words = set(normalized_chunk.split())
                            seen_words = set(seen_chunk.split())
                            if len(chunk_words) > 0 and len(seen_words) > 0:
                                intersection = len(chunk_words & seen_words)
                                union = len(chunk_words | seen_words)
                                similarity = intersection / union if union > 0 else 0
                                if similarity > 0.8:  # 80% similarity threshold
                                    is_duplicate = True
                                    break
                
                if not is_duplicate:
                    unique_chunks.append(chunk)
                    seen_chunks.add(normalized_chunk)
            
            # Use unique chunks only
            original_count = len(chunks)
            chunks = unique_chunks if unique_chunks else chunks
            removed_count = original_count - len(chunks)
            
            if removed_count > 0:
                print(f"Processed {len(chunks)} unique chunks (removed {removed_count} duplicate/near-duplicate chunks)")
            
            # Generate embeddings in batches for better performance
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = self.generate_embeddings(batch_chunks)
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
            
            if not all_embeddings or len(all_embeddings) != len(chunks):
                print(f"Failed to generate embeddings: got {len(all_embeddings)} for {len(chunks)} chunks")
                return False
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                point_id = str(uuid.uuid4())
                
                metadata_payload = {
                    'text': chunk,
                    'filename': filename,
                    'language': language,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'uploaded_at': timestamp,
                }
                metadata_payload.update(extra_metadata)

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata_payload
                )
                points.append(point)
            
            # Store points in Qdrant or memory
            if self.use_memory or self.qdrant_client is None:
                # Store in memory
                for point in points:
                    self.memory_storage.append({
                        'id': point.id,
                        'vector': point.vector,
                        'payload': point.payload
                    })
                print(f"Successfully stored {len(points)} chunks in memory for {filename}")
            else:
                # Store in Qdrant in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch_points = points[i:i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                print(f"Successfully stored {len(points)} chunks for {filename}")
            
            return True
            
        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[Union[str, List[str]]] = None,
        filenames: Optional[List[str]] = None,
        date_range: Optional[Tuple[float, float]] = None,
        diversity_lambda: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query with MMR (Maximal Marginal Relevance) for diversity.
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve (default: 5 for better context)
            language: Optional language filter(s) - string or list of strings
            filenames: Optional list of filenames to restrict retrieval
            date_range: Optional (start_ts, end_ts) tuple filtering by upload time
            diversity_lambda: Diversity parameter (0.0 = relevance only, 1.0 = diversity only, default: 0.7)
            
        Returns:
            List[Dict[str, Any]]: List of relevant chunks with metadata (diverse and non-repetitive)
        """
        try:
            language_filters: Optional[List[str]] = None
            if isinstance(language, str):
                language_filters = [language]
            elif isinstance(language, list):
                language_filters = language
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query], is_query=True)
            
            if not query_embedding:
                return []
            
            # Increase top_k for initial retrieval to allow for MMR diversification
            retrieval_k = max(top_k * 3, 20)
            
            if self.use_memory or self.qdrant_client is None:
                # Search in memory storage
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Filter by language if specified
                filtered_storage = self.memory_storage
                if language_filters:
                    filtered_storage = [
                        item for item in filtered_storage
                        if item['payload'].get('language') in language_filters
                    ]
                if filenames:
                    filtered_storage = [
                        item for item in filtered_storage
                        if item['payload'].get('filename') in filenames
                    ]
                if date_range:
                    start_ts, end_ts = date_range
                    filtered_storage = [
                        item for item in filtered_storage
                        if (
                            item['payload'].get('uploaded_at') is not None
                            and start_ts <= float(item['payload'].get('uploaded_at')) <= end_ts
                        )
                    ]
                
                if not filtered_storage:
                    return []
                
                # Calculate similarities
                query_vector = np.array(query_embedding[0]).reshape(1, -1)
                similarities = []
                
                for item in filtered_storage:
                    item_vector = np.array(item['vector']).reshape(1, -1)
                    similarity = cosine_similarity(query_vector, item_vector)[0][0]
                    similarities.append((similarity, item))
                
                # Sort by similarity and get top results
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_results = similarities[:retrieval_k]
                
                # Filter out very low similarity scores and duplicates
                candidate_results = []
                seen_texts = set()
                
                for score, item in top_results:
                    if score < 0.3:  # Filter out very low similarity
                        continue
                    
                    # Check for duplicate or near-duplicate text
                    chunk_text = item['payload']['text'].strip()
                    normalized_text = re.sub(r'\s+', ' ', chunk_text.lower())
                    
                    # Skip if we've seen very similar text (exact or near-exact match)
                    if normalized_text in seen_texts:
                        continue
                    
                    # Check for high similarity with existing candidates
                    is_duplicate = False
                    for seen_text in seen_texts:
                        # Simple similarity check: if texts are 90%+ similar, skip
                        if len(normalized_text) > 50 and len(seen_text) > 50:
                            # Word-level similarity
                            text_words = set(normalized_text.split())
                            seen_words = set(seen_text.split())
                            if len(text_words) > 0 and len(seen_words) > 0:
                                intersection = len(text_words & seen_words)
                                union = len(text_words | seen_words)
                                similarity = intersection / union if union > 0 else 0
                                if similarity > 0.9:  # 90% similarity threshold
                                    is_duplicate = True
                                    break
                    
                    if is_duplicate:
                        continue
                    
                    seen_texts.add(normalized_text)
                    
                    # Boost score slightly for longer chunks (more informative)
                    chunk_length = item['payload'].get('chunk_length', 0)
                    length_boost = min(chunk_length / 1000, 0.1)
                    adjusted_score = score + length_boost
                    
                    candidate_results.append((adjusted_score, item, item_vector))
                
                # Apply MMR (Maximal Marginal Relevance) for diversity
                if len(candidate_results) > 0:
                    # Sort by relevance score
                    candidate_results.sort(key=lambda x: x[0], reverse=True)
                    
                    # MMR selection
                    selected = []
                    selected_vectors = []
                    remaining = candidate_results.copy()
                    
                    # Always add the most relevant first
                    if remaining:
                        best_score, best_item, best_vector = remaining.pop(0)
                        selected.append((best_score, best_item))
                        selected_vectors.append(best_vector)
                    
                    # Select remaining items using MMR
                    while len(selected) < top_k and remaining:
                        best_mmr_score = -1
                        best_mmr_idx = -1
                        
                        for idx, (score, item, vector) in enumerate(remaining):
                            # Relevance to query
                            relevance = score
                            
                            # Maximum similarity to already selected items
                            max_similarity = 0.0
                            if selected_vectors:
                                for selected_vector in selected_vectors:
                                    similarity = cosine_similarity(
                                        vector.reshape(1, -1),
                                        selected_vector.reshape(1, -1)
                                    )[0][0]
                                    max_similarity = max(max_similarity, similarity)
                            
                            # MMR score: balance relevance and diversity
                            mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                            
                            if mmr_score > best_mmr_score:
                                best_mmr_score = mmr_score
                                best_mmr_idx = idx
                        
                        if best_mmr_idx >= 0:
                            score, item, vector = remaining.pop(best_mmr_idx)
                            selected.append((score, item))
                            selected_vectors.append(vector)
                        else:
                            break  # No good candidates left
                    
                    final_results = selected
                else:
                    final_results = []
                
                # Format results
                context_chunks = []
                for score, item in final_results:
                    context_chunks.append({
                        'text': item['payload']['text'],
                        'filename': item['payload']['filename'],
                        'language': item['payload']['language'],
                        'chunk_index': item['payload']['chunk_index'],
                        'score': float(score)
                    })
                
                return self._rerank_contexts(query, context_chunks, top_k)
            else:
                # Search in Qdrant
                search_params = {
                    'collection_name': self.collection_name,
                    'query_vector': query_embedding[0],
                    'limit': retrieval_k,
                    'with_payload': True,
                    'score_threshold': 0.3  # Minimum similarity threshold
                }
                
                filter_conditions: List[Any] = []
                if language_filters:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="language",
                            match=models.MatchAny(any=language_filters)
                        )
                    )
                if filenames:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchAny(any=filenames)
                        )
                    )
                if date_range:
                    start_ts, end_ts = date_range
                    filter_conditions.append(
                        models.FieldCondition(
                            key="uploaded_at",
                            range=models.Range(
                                gte=float(start_ts),
                                lte=float(end_ts)
                            )
                        )
                    )

                if filter_conditions:
                    search_params['query_filter'] = models.Filter(must=filter_conditions)
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(**search_params)
                
                # Deduplicate and apply MMR
                candidate_results = []
                seen_texts = set()
                
                for result in search_results:
                    chunk_text = result.payload['text'].strip()
                    normalized_text = re.sub(r'\s+', ' ', chunk_text.lower())
                    
                    # Skip duplicates
                    if normalized_text in seen_texts:
                        continue
                    
                    # Check for high similarity
                    is_duplicate = False
                    for seen_text in seen_texts:
                        if len(normalized_text) > 50 and len(seen_text) > 50:
                            text_words = set(normalized_text.split())
                            seen_words = set(seen_text.split())
                            if len(text_words) > 0 and len(seen_words) > 0:
                                intersection = len(text_words & seen_words)
                                union = len(text_words | seen_words)
                                similarity = intersection / union if union > 0 else 0
                                if similarity > 0.9:
                                    is_duplicate = True
                                    break
                    
                    if is_duplicate:
                        continue
                    
                    seen_texts.add(normalized_text)
                    
                    # Boost score for longer chunks
                    chunk_length = result.payload.get('chunk_length', 0)
                    length_boost = min(chunk_length / 1000, 0.1)
                    adjusted_score = result.score + length_boost
                    
                    # Get embedding for MMR
                    chunk_embedding = self.generate_embeddings([chunk_text])
                    if chunk_embedding:
                        candidate_results.append((adjusted_score, result, np.array(chunk_embedding[0])))
                
                # Apply MMR (similar to memory storage)
                if len(candidate_results) > 0:
                    from sklearn.metrics.pairwise import cosine_similarity
                    import numpy as np
                    
                    candidate_results.sort(key=lambda x: x[0], reverse=True)
                    
                    selected = []
                    selected_vectors = []
                    remaining = candidate_results.copy()
                    
                    if remaining:
                        best_score, best_result, best_vector = remaining.pop(0)
                        selected.append((best_score, best_result))
                        selected_vectors.append(best_vector.reshape(1, -1))
                    
                    while len(selected) < top_k and remaining:
                        best_mmr_score = -1
                        best_mmr_idx = -1
                        
                        for idx, (score, result, vector) in enumerate(remaining):
                            relevance = score
                            max_similarity = 0.0
                            
                            if selected_vectors:
                                for selected_vector in selected_vectors:
                                    similarity = cosine_similarity(
                                        vector.reshape(1, -1),
                                        selected_vector
                                    )[0][0]
                                    max_similarity = max(max_similarity, similarity)
                            
                            mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                            
                            if mmr_score > best_mmr_score:
                                best_mmr_score = mmr_score
                                best_mmr_idx = idx
                        
                        if best_mmr_idx >= 0:
                            score, result, vector = remaining.pop(best_mmr_idx)
                            selected.append((score, result))
                            selected_vectors.append(vector.reshape(1, -1))
                        else:
                            break
                    
                    final_results = selected
                else:
                    final_results = []
                
                # Format results
                context_chunks = []
                for adjusted_score, result in final_results:
                    context_chunks.append({
                        'text': result.payload['text'],
                        'filename': result.payload['filename'],
                        'language': result.payload['language'],
                        'chunk_index': result.payload['chunk_index'],
                        'score': float(adjusted_score)
                    })
                
                return self._rerank_contexts(query, context_chunks, top_k)
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status,
                'config': collection_info.config
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete all chunks belonging to a specific document.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Search for all points with the given filename
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=10000,  # Large limit to get all chunks
                with_payload=True
            )
            
            if search_results[0]:  # Check if any points found
                point_ids = [point.id for point in search_results[0]]
                
                # Delete the points
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                
                print(f"Deleted {len(point_ids)} chunks for {filename}")
                return True
            else:
                print(f"No chunks found for {filename}")
                return False
                
        except Exception as e:
            print(f"Error deleting document {filename}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            print(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
