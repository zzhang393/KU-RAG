"""
KU-RAG Main Entry Point

Unified interface for running the complete KU-RAG pipeline:
1. Build FAISS indices
2. Create meta-knowledge structure
3. Retrieve relevant knowledge
4. Generate visual passages
5. Evaluate results
"""

import argparse
import os
import sys
import json
from pathlib import Path


def build_index(args):
    """Build FAISS indices for a dataset."""
    print(f"\n{'='*60}")
    print(f"Building FAISS Index for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    dataset_path = f"datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{args.dataset}' not found!")
        return
    
    # Import and run create_faiss for the dataset
    sys.path.insert(0, dataset_path)
    try:
        import create_faiss
        print(f"✓ FAISS indices built successfully!")
    except Exception as e:
        print(f"✗ Error building indices: {e}")


def create_meta_knowledge(args):
    """Create meta-knowledge structure for a dataset."""
    print(f"\n{'='*60}")
    print(f"Creating Meta-Knowledge for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    dataset_path = f"datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{args.dataset}' not found!")
        return
    
    sys.path.insert(0, dataset_path)
    try:
        from meta_knowledge_manager import create_mk
        create_mk()
        print(f"✓ Meta-knowledge created successfully!")
    except Exception as e:
        print(f"✗ Error creating meta-knowledge: {e}")


def retrieve_knowledge(args):
    """Retrieve knowledge for questions."""
    print(f"\n{'='*60}")
    print(f"Retrieving Knowledge for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    dataset_path = f"datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{args.dataset}' not found!")
        return
    
    # Run the search_mk.py script
    os.chdir(dataset_path)
    sys.path.insert(0, '.')
    
    try:
        import search_mk
        # Run the main function if exists
        if hasattr(search_mk, '__main__'):
            exec(open('search_mk.py').read())
        print(f"✓ Knowledge retrieval completed!")
    except Exception as e:
        print(f"✗ Error during retrieval: {e}")
    finally:
        os.chdir('../..')


def generate_passages(args):
    """Generate visual passages."""
    print(f"\n{'='*60}")
    print(f"Generating Visual Passages for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    dataset_path = f"datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{args.dataset}' not found!")
        return
    
    os.chdir(dataset_path)
    sys.path.insert(0, '.')
    
    try:
        import passage_generator
        exec(open('passage_generator.py').read())
        print(f"✓ Visual passages generated!")
    except Exception as e:
        print(f"✗ Error generating passages: {e}")
    finally:
        os.chdir('../..')


def run_full_pipeline(args):
    """Run the complete pipeline."""
    print(f"\n{'='*60}")
    print(f"Running Full Pipeline for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    steps = [
        ("Build FAISS Index", build_index),
        ("Create Meta-Knowledge", create_meta_knowledge),
        ("Retrieve Knowledge", retrieve_knowledge),
        ("Generate Passages", generate_passages),
    ]
    
    for step_name, step_func in steps:
        print(f"\n[{step_name}]")
        if args.skip_existing and check_step_completed(args.dataset, step_name):
            print(f"  ⊙ Skipping (already completed)")
            continue
        
        try:
            step_func(args)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if not args.continue_on_error:
                print(f"\nPipeline stopped due to error.")
                return
    
    print(f"\n{'='*60}")
    print(f"Pipeline Completed!")
    print(f"{'='*60}\n")


def check_step_completed(dataset: str, step: str) -> bool:
    """Check if a pipeline step has been completed."""
    dataset_path = f"datasets/{dataset}"
    
    # Check for expected output files
    if step == "Build FAISS Index":
        # Check for .vec files
        return any(Path(dataset_path).glob("*.vec"))
    elif step == "Create Meta-Knowledge":
        # Check for mk directory
        return os.path.exists(f"{dataset_path}/mk/mk_data.json")
    elif step == "Retrieve Knowledge":
        # Check for results
        return os.path.exists(f"{dataset_path}/results/idx_result.json")
    elif step == "Generate Passages":
        # Check for generated images
        return os.path.exists(f"{dataset_path}/image_test") and \
               len(list(Path(f"{dataset_path}/image_test").glob("*.jpg"))) > 0
    
    return False


def evaluate(args):
    """Evaluate results using LLM"""
    print(f"\n{'='*60}")
    print(f"Evaluation for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    dataset_path = f"datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{args.dataset}' not found!")
        return
    
    # Import modules
    sys.path.insert(0, 'common')
    from llm_client import create_client
    from evaluator import create_evaluator, load_ground_truth
    
    # Load configuration
    config_file = f"{dataset_path}/llm_config.json"
    llm_config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
        print(f"Loaded LLM config: {config_file}")
    else:
        print("LLM config file not found, using default configuration")
        print("Hint: Please set environment variables LLM_API_KEY and LLM_API_URL")
    
    # Create LLM client
    client = create_client(llm_config)
    
    # Determine evaluation mode
    mode = getattr(args, 'eval_mode', 'kc_award')  # Default to KC-Award mode
    
    # Create answer save directory
    answer_dir = f"{dataset_path}/answers"
    os.makedirs(answer_dir, exist_ok=True)
    
    if mode == 'original':
        # Single image mode: only use original image
        print("\n[Mode: Single Image Evaluation]")
        evaluate_original_mode(args, dataset_path, client, answer_dir)
    else:
        # KC-Award mode: two-stage answer generation
        print("\n[Mode: KC-Award Prompt Evaluation]")
        evaluate_kc_award_mode(args, dataset_path, client, answer_dir)


def evaluate_original_mode(args, dataset_path: str, client, answer_dir: str):
    """Single image mode evaluation"""
    from evaluator import create_evaluator, load_ground_truth
    
    # Find original image directory
    image_dirs = ['image', 'val2014', 'images']
    original_image_dir = None
    for dir_name in image_dirs:
        test_dir = f"{dataset_path}/{dir_name}"
        if os.path.exists(test_dir):
            original_image_dir = test_dir
            break
    
    if not original_image_dir:
        print(f"Error: Original image directory not found")
        return
    
    print(f"Using original image directory: {original_image_dir}")
    
    # Load question data
    questions_list = load_questions(dataset_path, args.dataset)
    if not questions_list:
        print("Error: Unable to load question data")
        return
    
    print(f"Loaded {len(questions_list)} questions")
    
    # Build QA pairs list
    qa_pairs = []
    for question in questions_list:
        q_id = question.get('data_id') or question.get('question_id') or question.get('id')
        q_text = question.get('question')
        q_image_id = question.get('image_id')
        
        if not q_id or not q_text:
            continue
        
        # Find corresponding original image
        image_path = find_original_image(original_image_dir, q_id, q_image_id, args.dataset)
        
        if image_path and os.path.exists(image_path):
            qa_pairs.append({
                'id': q_id,
                'question': q_text,
                'image': image_path
            })
    
    print(f"Found {len(qa_pairs)} valid QA pairs")
    
    if len(qa_pairs) == 0:
        print("Error: No valid QA pairs found")
        return
    
    # Save path
    save_path = f"{answer_dir}/original_answers.json"
    
    # Batch get answers
    print(f"\nStarting LLM calls to get answers...")
    print(f"Results will be saved to: {save_path}")
    
    results = client.batch_get_answers(
        qa_pairs=qa_pairs,
        save_path=save_path,
        resume=True
    )
    
    # Evaluate results using evaluator
    print(f"\nStarting answer evaluation...")
    evaluator = create_evaluator(args.dataset)
    ground_truths = load_ground_truth(args.dataset, dataset_path)
    
    if not ground_truths:
        print(f"Warning: No ground truth data loaded. Skipping evaluation.")
        print(f"Please ensure ground truth files exist in {dataset_path}/qa_data/")
        return
    
    print(f"Loaded {len(ground_truths)} ground truth answers")
    
    # Format results to match evaluator expected format
    formatted_results = []
    for r in results:
        formatted_results.append({
            'id': r.get('id'),
            'question_id': r.get('id'),
            'question': r.get('question'),
            'answer': r.get('answer'),
            'image': r.get('image')
        })
    
    # Perform evaluation using evaluator
    eval_results, metrics = evaluator.evaluate_batch(formatted_results, ground_truths)
    
    # Save evaluation results using evaluator
    output_path = f"{answer_dir}/eval_original"
    evaluator.save_results(eval_results, metrics, output_path)
    
    print(f"\n{'='*60}")
    print(f"Single image mode evaluation completed!")
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Results saved to: {output_path}.xlsx")
    print(f"{'='*60}\n")


def evaluate_kc_award_mode(args, dataset_path: str, client, answer_dir: str):
    """KC-Award Prompt mode evaluation"""
    from evaluator import create_evaluator, load_ground_truth
    
    # Find original image directory
    image_dirs = ['image', 'val2014', 'images']
    original_image_dir = None
    for dir_name in image_dirs:
        test_dir = f"{dataset_path}/{dir_name}"
        if os.path.exists(test_dir):
            original_image_dir = test_dir
            break
    
    if not original_image_dir:
        print(f"Error: Original image directory not found")
        return
    
    # Find retrieved image directory
    retrieved_image_dirs = ['image_test', 'image_test2', 'generated_passages']
    retrieved_image_dir = None
    for dir_name in retrieved_image_dirs:
        test_dir = f"{dataset_path}/{dir_name}"
        if os.path.exists(test_dir):
            retrieved_image_dir = test_dir
            break
    
    if not retrieved_image_dir:
        print(f"Error: Retrieved image directory not found")
        print(f"Please run generate stage first to create visual passages")
        return
    
    print(f"Using original image directory: {original_image_dir}")
    print(f"Using retrieved image directory: {retrieved_image_dir}")
    
    # Load question data
    questions_list = load_questions(dataset_path, args.dataset)
    if not questions_list:
        print("Error: Unable to load question data")
        return
    
    print(f"Loaded {len(questions_list)} questions")
    
    # Build QA pairs list
    qa_pairs = []
    for question in questions_list:
        q_id = question.get('data_id') or question.get('question_id') or question.get('id')
        q_text = question.get('question')
        q_image_id = question.get('image_id')
        
        if not q_id or not q_text:
            continue
        
        # Find corresponding original image
        original_image = find_original_image(original_image_dir, q_id, q_image_id, args.dataset)
        
        # Find corresponding retrieved image
        retrieved_image = None
        for ext in ['.jpg', '.png', '.jpeg']:
            test_path = f"{retrieved_image_dir}/{q_id}{ext}"
            if os.path.exists(test_path):
                retrieved_image = test_path
                break
        
        if original_image and retrieved_image and os.path.exists(original_image):
            qa_pairs.append({
                'id': q_id,
                'question': q_text,
                'original_image': original_image,
                'retrieved_image': retrieved_image
            })
    
    print(f"Found {len(qa_pairs)} valid QA pairs")
    
    if len(qa_pairs) == 0:
        print("Error: No valid QA pairs found")
        return
    
    # Save paths
    original_save_path = f"{answer_dir}/original_answers.json"
    enhanced_save_path = f"{answer_dir}/enhanced_answers.json"
    
    # Batch get answers using KC-Award Prompt
    print(f"\nStarting LLM calls to get KC-Award Prompt answers...")
    print(f"Original answers will be saved to: {original_save_path}")
    print(f"Enhanced answers will be saved to: {enhanced_save_path}")
    
    original_results, enhanced_results = client.batch_get_kc_award_answers(
        qa_pairs=qa_pairs,
        original_save_path=original_save_path,
        enhanced_save_path=enhanced_save_path,
        resume=True
    )
    
    # Evaluate results using evaluator
    print(f"\nStarting answer evaluation...")
    evaluator = create_evaluator(args.dataset)
    ground_truths = load_ground_truth(args.dataset, dataset_path)
    
    if not ground_truths:
        print(f"Warning: No ground truth data loaded. Skipping evaluation.")
        print(f"Please ensure ground truth files exist in {dataset_path}/qa_data/")
        return
    
    print(f"Loaded {len(ground_truths)} ground truth answers")
    
    # Evaluate original answers using evaluator
    formatted_original = []
    for r in original_results:
        formatted_original.append({
            'id': r.get('id'),
            'question_id': r.get('id'),
            'question': r.get('question'),
            'answer': r.get('answer'),
            'image': r.get('image')
        })
    
    original_eval_results, original_metrics = evaluator.evaluate_batch(formatted_original, ground_truths)
    
    # Evaluate enhanced answers using evaluator
    formatted_enhanced = []
    for r in enhanced_results:
        formatted_enhanced.append({
            'id': r.get('id'),
            'question_id': r.get('id'),
            'question': r.get('question'),
            'answer': r.get('answer'),
            'original_image': r.get('original_image'),
            'retrieved_image': r.get('retrieved_image')
        })
    
    enhanced_eval_results, enhanced_metrics = evaluator.evaluate_batch(formatted_enhanced, ground_truths)
    
    # Save evaluation results using evaluator
    original_output_path = f"{answer_dir}/eval_original"
    enhanced_output_path = f"{answer_dir}/eval_enhanced"
    
    evaluator.save_results(original_eval_results, original_metrics, original_output_path)
    evaluator.save_results(enhanced_eval_results, enhanced_metrics, enhanced_output_path)
    
    print(f"\n{'='*60}")
    print(f"KC-Award Prompt mode evaluation completed!")
    print(f"Original answer accuracy: {original_metrics['accuracy']:.2%} ({original_metrics['correct']}/{original_metrics['total']})")
    print(f"Enhanced answer accuracy: {enhanced_metrics['accuracy']:.2%} ({enhanced_metrics['correct']}/{enhanced_metrics['total']})")
    print(f"Accuracy improvement: {(enhanced_metrics['accuracy'] - original_metrics['accuracy']):.2%}")
    print(f"\nResults saved:")
    print(f"  - Original: {original_output_path}.xlsx")
    print(f"  - Enhanced: {enhanced_output_path}.xlsx")
    print(f"{'='*60}\n")


def find_original_image(image_dir: str, q_id: str, image_id: str, dataset_name: str) -> str:
    """
    Find original image path
    
    Args:
        image_dir: Image directory
        q_id: Question ID
        image_id: Image ID
        dataset_name: Dataset name
    
    Returns:
        Image path
    """
    # Try different naming patterns
    possible_patterns = [
        f"{image_id}.jpg",
        f"{image_id}.png",
        f"{image_id}.jpeg",
        f"{q_id}.jpg",
        f"{q_id}.png",
    ]
    
    # Special handling for OK-VQA
    if 'okvqa' in dataset_name.lower() or 'ok-vqa' in dataset_name.lower():
        if image_id:
            possible_patterns.insert(0, f"COCO_val2014_{int(image_id):012d}.jpg")
    
    for pattern in possible_patterns:
        full_path = f"{image_dir}/{pattern}"
        if os.path.exists(full_path):
            return full_path
    
    return None


def load_questions(dataset_path: str, dataset_name: str) -> list:
    """Load question data"""
    questions = []
    
    # Try different file formats
    qa_data_dir = f"{dataset_path}/qa_data"
    
    # JSON format
    json_files = list(Path(qa_data_dir).glob("*.json")) if os.path.exists(qa_data_dir) else []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions.extend(data)
                elif isinstance(data, dict) and 'questions' in data:
                    questions.extend(data['questions'])
        except Exception as e:
            print(f"Failed to read {json_file}: {e}")
    
    # JSONL format
    jsonl_files = list(Path(qa_data_dir).glob("*.jsonl")) if os.path.exists(qa_data_dir) else []
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    questions.append(json.loads(line))
        except Exception as e:
            print(f"Failed to read {jsonl_file}: {e}")
    
    # CSV format
    csv_files = list(Path(dataset_path).glob("*.csv"))
    for csv_file in csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            questions.extend(df.to_dict('records'))
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
    
    return questions


def list_datasets():
    """List available datasets."""
    print("\nAvailable Datasets:")
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("  No datasets found!")
        return
    
    for dataset_dir in sorted(datasets_dir.iterdir()):
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            # Check if it has required files
            has_search = (dataset_dir / "search_mk.py").exists()
            has_faiss = (dataset_dir / "create_faiss.py").exists()
            status = "✓" if (has_search and has_faiss) else "✗"
            print(f"  {status} {dataset_dir.name}")


def main():
    parser = argparse.ArgumentParser(
        description="KU-RAG: Fine-Grained Knowledge Retrieval for VQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python main.py --list

  # Build FAISS index for OK-VQA
  python main.py --dataset okvqa --stage build_index

  # Run full pipeline for OVEN
  python main.py --dataset oven --stage full

  # Run retrieval only
  python main.py --dataset infoseek --stage retrieve

Stages:
  build_index      Build FAISS indices for images and text
  create_mk        Create meta-knowledge structure
  retrieve         Retrieve relevant knowledge units
  generate         Generate visual passages
  evaluate         Evaluate with LLM (GPT-4V, etc.)
  full             Run complete pipeline
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['okvqa', 'oven', 'infoseek', 'evqa'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=str,
        choices=['build_index', 'create_mk', 'retrieve', 'generate', 'evaluate', 'full'],
        default='full',
        help='Pipeline stage to run (default: full)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available datasets'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip steps that have already been completed'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue pipeline even if a step fails'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=['original', 'kc_award'],
        default='kc_award',
        help='Evaluation mode: original (single image) or kc_award (two-stage with KC-Award Prompt)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print(" "*15 + "KU-RAG Pipeline")
    print("="*60)
    
    # Handle list command
    if args.list:
        list_datasets()
        return
    
    # Validate dataset argument
    if not args.dataset:
        parser.print_help()
        print("\nError: --dataset is required (use --list to see available datasets)")
        sys.exit(1)
    
    # Set device environment variable
    os.environ['KURAG_DEVICE'] = args.device
    
    # Run appropriate stage
    stage_funcs = {
        'build_index': build_index,
        'create_mk': create_meta_knowledge,
        'retrieve': retrieve_knowledge,
        'generate': generate_passages,
        'evaluate': evaluate,
        'full': run_full_pipeline
    }
    
    stage_func = stage_funcs.get(args.stage)
    if stage_func:
        stage_func(args)
    else:
        print(f"Error: Unknown stage '{args.stage}'")
        sys.exit(1)
    
    print("\nDone!\n")


if __name__ == "__main__":
    main()

