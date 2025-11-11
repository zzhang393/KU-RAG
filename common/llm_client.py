"""
LLM Client for Visual Question Answering
Supports GPT-4V and other vision-language models
"""

import requests
import json
import base64
import time
import os
from typing import Optional, Dict, Any


class LLMClient:
    """General-purpose LLM client supporting visual question answering"""
    
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
    
    def get_answer(self, 
                   question: str, 
                   image_path: Optional[str] = None,
                   system_prompt: Optional[str] = None) -> str:
        """
        Call LLM to get answer
        
        Args:
            question: Question text
            image_path: Image path (optional)
            system_prompt: System prompt (optional)
        
        Returns:
            LLM answer text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build message content
        if image_path and os.path.exists(image_path):
            # Visual question answering mode
            image_base64 = self.encode_image(image_path)
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question},
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
        
        data = {
            'model': self.model,
            'messages': messages,
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
    
    def batch_get_answers(self, 
                         qa_pairs: list,
                         save_path: Optional[str] = None,
                         resume: bool = True) -> list:
        """
        Batch get answers
        
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

