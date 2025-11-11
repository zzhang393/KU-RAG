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
    """Evaluate results (placeholder for VLM integration)."""
    print(f"\n{'='*60}")
    print(f"Evaluation for {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    print("Note: This is a placeholder for VLM-based evaluation.")
    print("To complete evaluation:")
    print("  1. Feed generated passages to your VLM (e.g., GPT-4V, LLaVA)")
    print("  2. Collect answers")
    print("  3. Compare with ground truth")
    print("\nFor reference implementations, see:")
    print(f"  - datasets/{args.dataset}/answer/")
    print("  - External VLM inference scripts")


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
  evaluate         Evaluate with VLM (placeholder)
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

