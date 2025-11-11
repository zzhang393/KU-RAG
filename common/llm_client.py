"""
LLM Client for Visual Question Answering
Supports GPT-4V and other vision-language models
"""

import requests
import json
import base64
import time
import os
from typing import Optional, Dict, Any, List


class LLMClient:
    """General-purpose LLM client supporting visual question answering"""
    
    # KC-Award Prompt: Default prompt for two-stage answer generation
    KC_AWARD_PROMPT = (
        "You have already generated an initial answer based on a previous question and image. "
        "Now, new image information is provided, which may or may not be related to the original content. "
        "Your task is to carefully compare and integrate this new information **while still considering the original image and question**. "
        "Do not ignore the original image or context. "
        "If the new input provides additional insights or contradicts the initial answer, revise your response accordingly. "
        "If the new information does not affect the original answer, simply return the same answer. "
        "Your answer should be concise."
    )
    
    def __init__(self, 
                 api_url: str = None,
                 api_key: str = None,
                 model: str = 'gpt-4-turbo',
                 max_tokens: int = 300,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        """
        Initialize LLM client
        
        Args:
            api_url: API service URL
            api_key: API key
            model: Model name
            max_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retries
            retry_delay: Retry delay in seconds
        """
        self.api_url = api_url or os.environ.get('LLM_API_URL', 'https://api.openai.com/v1/chat/completions')
        self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            print("Warning: API key not set. Please set it via parameter or environment variable LLM_API_KEY.")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _call_api(self, conversation_history: List[Dict]) -> str:
        """
        Call API to get answer (internal method)
        
        Args:
            conversation_history: Conversation history
        
        Returns:
            LLM answer text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            'model': self.model,
            'messages': conversation_history,
            'max_tokens': self.max_tokens
        }
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                answer = result['choices'][0]['message']['content']
                return answer.strip()
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"Failed to parse response: {e}")
                return ""
        
        print(f"Reached maximum retry attempts, unable to get answer")
        return ""
    
    def get_answer(self, 
                   question: str, 
                   image_path: Optional[str] = None,
                   system_prompt: Optional[str] = None) -> str:
        """
        Call LLM to get answer (single image, single round)
        
        Args:
            question: Question text
            image_path: Image path (optional)
            system_prompt: System prompt (optional)
        
        Returns:
            LLM answer text
        """
        # Build message content
        if image_path and os.path.exists(image_path):
            # Visual question answering mode
            image_base64 = self.encode_image(image_path)
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question + " Your answer should be concise."},
                    {'type': 'image_url', 
                     'image_url': {'url': f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
        else:
            # Text-only mode
            messages = [{'role': 'user', 'content': question}]
        
        # Add system prompt
        if system_prompt:
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        
        return self._call_api(messages)
    
    def get_kc_award_answer(self,
                            question: str,
                            original_image_path: str,
                            retrieved_image_path: str,
                            original_answer: Optional[str] = None,
                            kc_award_prompt: Optional[str] = None) -> tuple:
        """
        Get answer using KC-Award Prompt (two-stage answer generation)
        
        Stage 1: Answer question using original image
        Stage 2: Integrate retrieved image information using KC-Award Prompt and update answer
        
        Args:
            question: Question text
            original_image_path: Original image path
            retrieved_image_path: Retrieved knowledge image path
            original_answer: Original answer (if provided, skip stage 1)
            kc_award_prompt: KC-Award Prompt text (optional, uses default if not provided)
        
        Returns:
            (original_answer, enhanced_answer): Tuple of original answer and enhanced answer
        """
        # Stage 1: Get original answer
        if original_answer is None:
            base64_image1 = self.encode_image(original_image_path)
            conversation_history = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question + " Your answer should be concise."},
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image1}"}}
                ]
            }]
            original_answer = self._call_api(conversation_history)
        else:
            # If original answer is provided, build conversation history
            base64_image1 = self.encode_image(original_image_path)
            conversation_history = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question + " Your answer should be concise."},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image1}"}}
                    ]
                },
                {'role': 'assistant', 'content': original_answer}
            ]
        
        # Stage 2: Update answer using retrieved image with KC-Award Prompt
        if not isinstance(conversation_history, list):
            base64_image1 = self.encode_image(original_image_path)
            conversation_history = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image1}"}}
                    ]
                },
                {'role': 'assistant', 'content': original_answer}
            ]
        
        # Add retrieved image with KC-Award Prompt
        base64_image2 = self.encode_image(retrieved_image_path)
        prompt_text = kc_award_prompt or self.KC_AWARD_PROMPT
        
        conversation_history.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt_text},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image2}"}}
            ]
        })
        
        enhanced_answer = self._call_api(conversation_history)
        
        return original_answer, enhanced_answer
    
    def batch_get_answers(self, 
                         qa_pairs: list,
                         save_path: Optional[str] = None,
                         resume: bool = True) -> list:
        """
        Batch get answers (single image mode)
        
        Args:
            qa_pairs: List of QA pairs, format: [{'question': ..., 'image': ..., 'id': ...}, ...]
            save_path: Save path (optional)
            resume: Whether to resume from checkpoint
        
        Returns:
            List of answers
        """
        results = []
        
        # Load existing results if resuming from checkpoint
        if resume and save_path and os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Resuming from checkpoint, {len(results)} completed")
        
        completed_ids = {r.get('id') or r.get('question_id') for r in results}
        
        # Process remaining QA pairs
        for i, qa in enumerate(qa_pairs):
            qa_id = qa.get('id') or qa.get('question_id') or qa.get('data_id')
            
            # Skip already completed
            if qa_id in completed_ids:
                continue
            
            question = qa['question']
            image_path = qa.get('image')
            
            print(f"\n[{i + 1}/{len(qa_pairs)}] ID: {qa_id}")
            print(f"Question: {question}")
            
            start_time = time.time()
            answer = self.get_answer(question, image_path)
            elapsed = time.time() - start_time
            
            print(f"Answer: {answer}")
            print(f"Time elapsed: {elapsed:.2f}s")
            
            result = {
                'id': qa_id,
                'question': question,
                'answer': answer,
                'image': image_path,
                'elapsed_time': elapsed
            }
            
            # Preserve other fields from original data
            for key in qa:
                if key not in result:
                    result[key] = qa[key]
            
            results.append(result)
            
            # Periodically save
            if save_path and (len(results) % 10 == 0 or i == len(qa_pairs) - 1):
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(results)} results to {save_path}")
        
        return results
    
    def batch_get_kc_award_answers(self,
                                  qa_pairs: list,
                                  original_save_path: str,
                                  enhanced_save_path: str,
                                  resume: bool = True) -> tuple:
        """
        Batch get answers using KC-Award Prompt (two-stage mode)
        
        Args:
            qa_pairs: List of QA pairs, format: [{'question': ..., 'original_image': ..., 'retrieved_image': ..., 'id': ...}, ...]
            original_save_path: Original answer save path
            enhanced_save_path: Enhanced answer save path
            resume: Whether to resume from checkpoint
        
        Returns:
            (original_results, enhanced_results): Tuple of original answer list and enhanced answer list
        """
        original_results = []
        enhanced_results = []
        
        # Load existing original answers
        if resume and os.path.exists(original_save_path):
            with open(original_save_path, 'r', encoding='utf-8') as f:
                original_results = json.load(f)
            print(f"Resuming from checkpoint, {len(original_results)} original answers completed")
        
        # Load existing enhanced answers
        if resume and os.path.exists(enhanced_save_path):
            with open(enhanced_save_path, 'r', encoding='utf-8') as f:
                enhanced_results = json.load(f)
            print(f"Resuming from checkpoint, {len(enhanced_results)} enhanced answers completed")
        
        completed_ids = {r.get('id') or r.get('question_id') for r in enhanced_results}
        
        # Build dictionary of existing answers
        original_dict = {}
        for r in original_results:
            r_id = r.get('id') or r.get('question_id')
            if r_id:
                original_dict[r_id] = r.get('answer', '')
        
        # Process remaining QA pairs
        for i, qa in enumerate(qa_pairs):
            qa_id = qa.get('id') or qa.get('question_id') or qa.get('data_id')
            
            # Skip already completed
            if qa_id in completed_ids:
                continue
            
            question = qa['question']
            original_image = qa.get('original_image') or qa.get('image')
            retrieved_image = qa.get('retrieved_image') or qa.get('rag_image')
            
            if not retrieved_image:
                print(f"Warning: ID {qa_id} missing retrieved image, skipping")
                continue
            
            print(f"\n[{i + 1}/{len(qa_pairs)}] ID: {qa_id}")
            print(f"Question: {question}")
            
            start_time = time.time()
            
            # Check if original answer already exists
            original_answer = original_dict.get(qa_id)
            
            # Call KC-Award Prompt answer generation
            original_answer, enhanced_answer = self.get_kc_award_answer(
                question=question,
                original_image_path=original_image,
                retrieved_image_path=retrieved_image,
                original_answer=original_answer
            )
            
            elapsed = time.time() - start_time
            
            print(f"Original answer: {original_answer}")
            print(f"Enhanced answer: {enhanced_answer}")
            print(f"Time elapsed: {elapsed:.2f}s")
            
            # Build results
            original_result = {
                'id': qa_id,
                'question': question,
                'answer': original_answer,
                'image': original_image,
                'elapsed_time': elapsed
            }
            
            enhanced_result = {
                'id': qa_id,
                'question': question,
                'answer': enhanced_answer,
                'original_image': original_image,
                'retrieved_image': retrieved_image,
                'elapsed_time': elapsed
            }
            
            # Preserve other fields from original data
            for key in qa:
                if key not in original_result and key not in ['rag_image', 'retrieved_image']:
                    original_result[key] = qa[key]
                if key not in enhanced_result:
                    enhanced_result[key] = qa[key]
            
            # Update original answer (if not previously existed)
            if qa_id not in original_dict:
                original_results.append(original_result)
            
            enhanced_results.append(enhanced_result)
            
            # Periodically save
            if len(enhanced_results) % 10 == 0 or i == len(qa_pairs) - 1:
                with open(original_save_path, 'w', encoding='utf-8') as f:
                    json.dump(original_results, f, indent=2, ensure_ascii=False)
                with open(enhanced_save_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(original_results)} original answers and {len(enhanced_results)} enhanced answers")
        
        return original_results, enhanced_results


def create_client(config: Dict[str, Any] = None) -> LLMClient:
    """
    Factory function to create LLM client
    
    Args:
        config: Configuration dictionary
    
    Returns:
        LLMClient instance
    """
    if config is None:
        config = {}
    
    return LLMClient(
        api_url=config.get('api_url'),
        api_key=config.get('api_key'),
        model=config.get('model', 'gpt-4-turbo'),
        max_tokens=config.get('max_tokens', 300),
        max_retries=config.get('max_retries', 3),
        retry_delay=config.get('retry_delay', 5)
    )

