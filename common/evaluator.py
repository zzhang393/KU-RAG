"""
Evaluator Module
Supports answer evaluation and metric calculation for different datasets
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


class Evaluator:
    """Base class for answer evaluators"""
    
    def __init__(self, dataset_name: str):
        """
        Initialize evaluator
        
        Args:
            dataset_name: Dataset name
        """
        self.dataset_name = dataset_name
    
    def evaluate_single(self, predicted: str, ground_truth: Any) -> bool:
        """
        Evaluate a single answer
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
        
        Returns:
            Whether the answer is correct
        """
        raise NotImplementedError
    
    def evaluate_batch(self, 
                      predictions: List[Dict], 
                      ground_truths: Dict[str, Any]) -> tuple:
        """
        Batch evaluation
        
        Args:
            predictions: List of prediction results
            ground_truths: Dictionary of ground truth answers
        
        Returns:
            (results, metrics): Evaluation results and metrics
        """
        results = []
        correct = 0
        total = 0
        
        for pred in predictions:
            pred_id = pred.get('id') or pred.get('question_id')
            pred_answer = pred.get('answer', '').lower()
            
            if pred_id not in ground_truths:
                continue
            
            gt = ground_truths[pred_id]
            is_correct = self.evaluate_single(pred_answer, gt)
            
            result = pred.copy()
            result['correct'] = is_correct
            result['ground_truth'] = gt
            results.append(result)
            
            if is_correct:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        return results, metrics
    
    def save_results(self, 
                    results: List[Dict], 
                    metrics: Dict,
                    output_path: str):
        """
        Save evaluation results
        
        Args:
            results: Evaluation results
            metrics: Evaluation metrics
            output_path: Output path (without extension)
        """
        # Save as Excel
        df = pd.DataFrame(results)
        df.to_excel(f"{output_path}.xlsx", index=False)
        
        # Save as JSON
        with open(f"{output_path}.json", 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Save wrong cases
        wrong_results = [r for r in results if not r.get('correct', False)]
        wrong_ids = [r.get('id') or r.get('question_id') for r in wrong_results]
        
        with open(f"{output_path}_wrong.json", 'w', encoding='utf-8') as f:
            json.dump(wrong_ids, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation results saved:")
        print(f"  - Excel: {output_path}.xlsx")
        print(f"  - JSON: {output_path}.json")
        print(f"  - Wrong cases list: {output_path}_wrong.json")


class OVENEvaluator(Evaluator):
    """OVEN dataset evaluator"""
    
    def evaluate_single(self, predicted: str, ground_truth: Any) -> bool:
        """
        OVEN: Simple string containment check
        
        Args:
            predicted: Predicted answer (lowercased)
            ground_truth: Ground truth answer (string)
        
        Returns:
            Whether the answer is correct
        """
        if isinstance(ground_truth, str):
            gt_lower = ground_truth.lower()
            return gt_lower in predicted
        return False


class InfoSeekEvaluator(Evaluator):
    """InfoSeek dataset evaluator"""
    
    def evaluate_single(self, predicted: str, ground_truth: Any) -> bool:
        """
        InfoSeek: Ground truth is a list, check if any answer is in prediction
        
        Args:
            predicted: Predicted answer (lowercased)
            ground_truth: Ground truth answer (list)
        
        Returns:
            Whether the answer is correct
        """
        if isinstance(ground_truth, list):
            for ans in ground_truth:
                if ans.lower() in predicted:
                    return True
        return False


class OKVQAEvaluator(Evaluator):
    """OK-VQA dataset evaluator"""
    
    def evaluate_single(self, predicted: str, ground_truth: Any) -> bool:
        """
        OK-VQA: Ground truth is a set, check if any answer is in prediction
        
        Args:
            predicted: Predicted answer (lowercased)
            ground_truth: Ground truth answer (set or list)
        
        Returns:
            Whether the answer is correct
        """
        if isinstance(ground_truth, (set, list)):
            for ans in ground_truth:
                if str(ans).lower() in predicted:
                    return True
        return False


class EVQAEvaluator(Evaluator):
    """E-VQA dataset evaluator"""
    
    def evaluate_single(self, predicted: str, ground_truth: Any) -> bool:
        """
        E-VQA: Simple string containment check
        
        Args:
            predicted: Predicted answer (lowercased)
            ground_truth: Ground truth answer (string)
        
        Returns:
            Whether the answer is correct
        """
        if isinstance(ground_truth, str):
            gt_lower = ground_truth.lower()
            return gt_lower in predicted
        return False


def create_evaluator(dataset_name: str) -> Evaluator:
    """
    Factory function to create evaluator
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        Corresponding evaluator instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if 'oven' in dataset_name_lower:
        return OVENEvaluator(dataset_name)
    elif 'infoseek' in dataset_name_lower or 'info' in dataset_name_lower:
        return InfoSeekEvaluator(dataset_name)
    elif 'okvqa' in dataset_name_lower or 'ok-vqa' in dataset_name_lower:
        return OKVQAEvaluator(dataset_name)
    elif 'evqa' in dataset_name_lower or 'e-vqa' in dataset_name_lower:
        return EVQAEvaluator(dataset_name)
    else:
        # Default to simple string containment
        print(f"Warning: Unknown dataset '{dataset_name}', using default evaluator")
        return OVENEvaluator(dataset_name)


def load_ground_truth(dataset_name: str, dataset_path: str) -> Dict[str, Any]:
    """
    Load ground truth answers
    
    Args:
        dataset_name: Dataset name
        dataset_path: Dataset path
    
    Returns:
        Ground truth dictionary {id: answer}
    """
    dataset_name_lower = dataset_name.lower()
    ground_truths = {}
    
    if 'oven' in dataset_name_lower:
        # OVEN dataset
        qa_file = Path(dataset_path) / 'qa_data' / 'oven_entity_test.jsonl'
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    ground_truths[data['data_id']] = data['entity_text']
    
    elif 'infoseek' in dataset_name_lower:
        # InfoSeek dataset
        qa_file = Path(dataset_path) / 'qa_data' / 'infoseek_test.jsonl'
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    ground_truths[data['data_id']] = data['answer']
    
    elif 'okvqa' in dataset_name_lower or 'ok-vqa' in dataset_name_lower:
        # OK-VQA dataset
        qa_file = Path(dataset_path) / 'qa_data' / 'OpenEnded_mscoco_val2014_questions.json'
        ans_file = Path(dataset_path) / 'qa_data' / 'mscoco_val2014_annotations.json'
        
        if qa_file.exists() and ans_file.exists():
            with open(ans_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            for annotation in annotations['annotations']:
                q_id = annotation['question_id']
                # Only keep answers with confidence="yes"
                answers = set()
                for ans_data in annotation['answers']:
                    if ans_data.get('answer_confidence') == 'yes':
                        answers.add(ans_data['answer'])
                ground_truths[str(q_id)] = answers
    
    elif 'evqa' in dataset_name_lower or 'e-vqa' in dataset_name_lower:
        # E-VQA dataset
        csv_file = Path(dataset_path) / 'E-VQA_data.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                q_id = row.get('question_id') or row.get('id')
                answer = row.get('answer') or row.get('entity_text')
                if q_id is not None and answer is not None:
                    ground_truths[str(q_id)] = str(answer)
    
    return ground_truths
