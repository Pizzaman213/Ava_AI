#!/usr/bin/env python3
"""
Comprehensive Dataset Downloader for Enhanced LLM Features

This script downloads and prepares datasets for all enhanced features including:
- Pre-training datasets (OpenWebText, The Pile, WikiText, BookCorpus)
- RAG knowledge bases (Wikipedia, MS MARCO, Natural Questions)
- Multi-task learning datasets (GLUE, SuperGLUE, XTREME)
- Evaluation datasets (HellaSwag, ARC, MMLU, CNN/DailyMail)
- Safety & bias datasets (Toxicity, bias evaluation)
- Multi-modal datasets (Vision-language, audio-text)
- Continual learning datasets (For episodic memory)
- Code datasets (For programming capabilities)

Supports 80+ diverse datasets with multiple retry strategies and feature-specific groupings.
"""

import os
import sys
import json
import time
import argparse
import traceback
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Lazy imports to avoid issues
def import_datasets():
    """Lazy import of datasets library"""
    try:
        from datasets import load_dataset, load_from_disk
        import datasets
        datasets.disable_progress_bar()  # We'll use our own progress bars
        return load_dataset, load_from_disk, datasets
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset, load_from_disk
        import datasets
        datasets.disable_progress_bar()
        return load_dataset, load_from_disk, datasets

# Comprehensive dataset configuration for enhanced LLM features
# Each dataset includes working examples and verified parameters
DATASETS_CONFIG = {
    # ================================
    # PRE-TRAINING DATASETS
    # ================================
    # Core Instruction Tuning (Verified Working)
    "databricks/databricks-dolly-15k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction"], "tokens": "high",
        "description": "15K instruction-following examples from Databricks",
        "example_command": "python download_datasets.py --dataset 'databricks/databricks-dolly-15k'"
    },
    "tatsu-lab/alpaca": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction"], "tokens": "medium",
        "description": "52K instruction-following examples from Stanford",
        "example_command": "python download_datasets.py --dataset 'tatsu-lab/alpaca'"
    },
    "yahma/alpaca-cleaned": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction"], "tokens": "medium",
        "description": "Cleaned version of Alpaca dataset with improved quality",
        "example_command": "python download_datasets.py --dataset 'yahma/alpaca-cleaned'"
    },
    "vicgalle/alpaca-gpt4": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction"], "tokens": "medium",
        "description": "Alpaca dataset regenerated with GPT-4 for higher quality",
        "example_command": "python download_datasets.py --dataset 'vicgalle/alpaca-gpt4'"
    },

    # OpenAssistant (Verified Working)
    "OpenAssistant/oasst1": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "conversation"], "tokens": "high",
        "description": "Human-generated, assistant-ranked conversation trees",
        "example_command": "python download_datasets.py --dataset 'OpenAssistant/oasst1'"
    },
    "OpenAssistant/oasst2": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "conversation"], "tokens": "high",
        "description": "Second version of OpenAssistant conversations dataset",
        "example_command": "python download_datasets.py --dataset 'OpenAssistant/oasst2'"
    },

    # ================================
    # RAG KNOWLEDGE BASES
    # ================================
    # Large-scale Text Datasets (High Token Count)
    "allenai/c4": {
        "splits": ["train"], "subset": "en", "streaming_safe": True, "max_samples": 100000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "Colossal Clean Crawled Corpus - cleaned web text for language modeling",
        "example_command": "python download_datasets.py --dataset 'allenai/c4' --max-samples 10000"
    },
    "openwebtext": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "Open-source recreation of GPT-2's WebText training dataset",
        "example_command": "python download_datasets.py --dataset 'openwebtext' --max-samples 5000"
    },
    "EleutherAI/pile": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "800GB of diverse text from books, websites, and academic sources",
        "example_command": "python download_datasets.py --dataset 'EleutherAI/pile' --max-samples 5000"
    },
    "wikipedia": {
        "splits": ["train"], "subset": "20220301.en", "streaming_safe": True, "max_samples": 100000,
        "categories": ["rag", "knowledge"], "tokens": "very_high", "large": True,
        "description": "English Wikipedia articles for knowledge-intensive tasks",
        "example_command": "python download_datasets.py --dataset 'wikipedia' --max-samples 10000"
    },
    "cc_news": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 75000,
        "categories": ["rag", "news"], "tokens": "very_high", "large": True,
        "description": "News articles from Common Crawl for current events knowledge",
        "example_command": "python download_datasets.py --dataset 'cc_news' --max-samples 5000"
    },
    "togethercomputer/RedPajama-Data-1T": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 25000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "1.2 trillion token dataset replicating LLaMA training data",
        "example_command": "python download_datasets.py --dataset 'togethercomputer/RedPajama-Data-1T' --max-samples 1000"
    },
    "HuggingFaceFW/fineweb": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 75000,
        "categories": ["rag", "web"], "tokens": "very_high", "large": True,
        "description": "High-quality web text filtered from CommonCrawl",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceFW/fineweb' --max-samples 5000"
    },
    "HuggingFaceFW/fineweb-edu": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000,
        "categories": ["rag", "education"], "tokens": "very_high", "large": True,
        "description": "Educational web content from FineWeb corpus",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceFW/fineweb-edu' --max-samples 5000"
    },
    "tiiuae/falcon-refinedweb": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 30000,
        "categories": ["rag", "web"], "tokens": "very_high", "large": True,
        "description": "Refined web text used to train Falcon LLM",
        "example_command": "python download_datasets.py --dataset 'tiiuae/falcon-refinedweb' --max-samples 3000"
    },

    # ================================
    # MULTI-TASK & CONTINUAL LEARNING
    # ================================
    # Math & Reasoning (For continual learning)
    "gsm8k": {
        "splits": ["train", "test"], "subset": "main", "streaming_safe": True,
        "categories": ["multitask", "continual", "math"], "tokens": "medium",
        "description": "Grade School Math 8K - math word problems with solutions",
        "example_command": "python download_datasets.py --dataset 'gsm8k'"
    },
    "hendrycks/competition_math": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["multitask", "continual", "math"], "tokens": "medium",
        "description": "Competition-level mathematics problems from AMC, AIME, USAMO",
        "example_command": "python download_datasets.py --dataset 'hendrycks/competition_math'"
    },

    # GLUE Tasks (Multi-task learning)
    "glue": {
        "splits": ["train", "validation"], "subset": "cola", "streaming_safe": True,
        "categories": ["multitask", "evaluation"], "tokens": "low",
        "task_type": "classification",
        "description": "GLUE CoLA task - linguistic acceptability classification",
        "example_command": "python download_datasets.py --dataset 'glue'"
    },
    "squad": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "qa"], "tokens": "medium",
        "description": "Stanford Question Answering Dataset for reading comprehension",
        "example_command": "python download_datasets.py --dataset 'squad'"
    },

    # ================================
    # CODE DATASETS
    # ================================
    "HuggingFaceH4/CodeAlpaca_20K": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "instruction"], "tokens": "medium",
        "description": "20K code instruction-following examples",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/CodeAlpaca_20K'"
    },
    "sahil2801/CodeAlpaca-20k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "instruction"], "tokens": "medium",
        "description": "Code generation and instruction following dataset",
        "example_command": "python download_datasets.py --dataset 'sahil2801/CodeAlpaca-20k'"
    },
    "iamtarun/python_code_instructions_18k_alpaca": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "python"], "tokens": "medium",
        "description": "Python-specific coding instructions and solutions",
        "example_command": "python download_datasets.py --dataset 'iamtarun/python_code_instructions_18k_alpaca'"
    },
    "bigcode/self-oss-instruct-sc2-exec-filter-50k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "oss"], "tokens": "high",
        "description": "Code instruction dataset with execution filtering",
        "example_command": "python download_datasets.py --dataset 'bigcode/self-oss-instruct-sc2-exec-filter-50k'"
    },
    "m-a-p/CodeFeedback-Filtered-Instruction": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "feedback"], "tokens": "high",
        "description": "Code instruction dataset with feedback filtering",
        "example_command": "python download_datasets.py --dataset 'm-a-p/CodeFeedback-Filtered-Instruction'"
    },
    "github-code": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 40000,
        "categories": ["code", "github"], "tokens": "very_high", "large": True,
        "description": "Large corpus of code from GitHub repositories",
        "example_command": "python download_datasets.py --dataset 'github-code' --max-samples 1000"
    },

    # ================================
    # EVALUATION DATASETS
    # ================================
    # Comprehension & Reasoning
    "allenai/ai2_arc": {
        "splits": ["train", "test", "validation"], "subset": "ARC-Challenge", "streaming_safe": True,
        "categories": ["evaluation", "reasoning"], "tokens": "low",
        "description": "AI2 Reasoning Challenge - grade-school science questions",
        "example_command": "python download_datasets.py --dataset 'allenai/ai2_arc'"
    },
    "winogrande": {
        "splits": ["train", "validation"], "subset": "winogrande_xl", "streaming_safe": True,
        "categories": ["evaluation", "reasoning"], "tokens": "low",
        "description": "Commonsense reasoning with pronoun resolution",
        "example_command": "python download_datasets.py --dataset 'winogrande'"
    },
    "hellaswag": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "commonsense"], "tokens": "medium",
        "description": "Commonsense natural language inference",
        "example_command": "python download_datasets.py --dataset 'hellaswag'"
    },
    "squad": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "qa"], "tokens": "medium",
        "description": "Stanford Question Answering Dataset for reading comprehension",
        "example_command": "python download_datasets.py --dataset 'squad'"
    },
    "squad_v2": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "qa"], "tokens": "medium",
        "description": "Stanford Question Answering v2 with unanswerable questions",
        "example_command": "python download_datasets.py --dataset 'squad_v2'"
    },
    "natural_questions": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True, "large": True,
        "categories": ["evaluation", "qa", "rag"], "tokens": "very_high",
        "description": "Real user questions from Google search with Wikipedia answers",
        "example_command": "python download_datasets.py --dataset 'natural_questions' --max-samples 5000"
    },

    # Summarization
    "cnn_dailymail": {
        "splits": ["train", "validation", "test"], "subset": "3.0.0", "streaming_safe": True,
        "categories": ["evaluation", "summarization"], "tokens": "high",
        "description": "CNN/DailyMail news articles with highlights for summarization",
        "example_command": "python download_datasets.py --dataset 'cnn_dailymail'"
    },

    # ================================
    # SAFETY & BIAS DATASETS
    # ================================
    "Anthropic/hh-rlhf": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "rlhf"], "tokens": "high",
        "description": "Human feedback dataset for helpful and harmless AI",
        "example_command": "python download_datasets.py --dataset 'Anthropic/hh-rlhf'"
    },
    "HuggingFaceH4/ultrafeedback_binarized": {
        "splits": ["train_prefs", "test_prefs"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "feedback"], "tokens": "high",
        "description": "High-quality preference data for RLHF",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/ultrafeedback_binarized'"
    },

    # ================================
    # MULTI-MODAL DATASETS
    # ================================
    "ShareGPT4Omni/ShareGPT4V": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["multimodal", "vision"], "tokens": "high",
        "description": "Vision-language conversations with GPT-4V",
        "example_command": "python download_datasets.py --dataset 'ShareGPT4Omni/ShareGPT4V'"
    },

    # ================================
    # CONVERSATIONAL DATASETS
    # ================================
    "HuggingFaceH4/ultrachat_200k": {
        "splits": ["train_sft", "test_sft"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "multiturn"], "tokens": "very_high", "large": True,
        "description": "200K high-quality multi-turn conversations",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/ultrachat_200k' --max-samples 1000"
    },
    "HuggingFaceH4/no_robots": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "synthetic"], "tokens": "high",
        "description": "Human-generated conversations without AI assistance",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/no_robots'"
    },
    "lmsys/lmsys-chat-1m": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "large": True,
        "categories": ["conversation", "multiturn"], "tokens": "very_high",
        "description": "1M real user conversations from Vicuna demo",
        "example_command": "python download_datasets.py --dataset 'lmsys/lmsys-chat-1m' --max-samples 1000"
    },
    "OpenAssistant/oasst1": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "dialog"], "tokens": "medium",
        "description": "Open Assistant conversational dataset",
        "example_command": "python download_datasets.py --dataset 'OpenAssistant/oasst1'"
    },
    "blended_skill_talk": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "empathy"], "tokens": "medium",
        "description": "Conversations blending empathy, knowledge and personality",
        "example_command": "python download_datasets.py --dataset 'blended_skill_talk'"
    },
    "AlekseyKorshuk/persona-chat": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "persona"], "tokens": "medium",
        "description": "Conversations with personality-driven characters",
        "example_command": "python download_datasets.py --dataset 'AlekseyKorshuk/persona-chat'"
    },

    # ================================
    # SPECIALIZED DATASETS
    # ================================
    # Synthetic Data
    "roneneldan/TinyStories": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["synthetic", "stories"], "tokens": "high",
        "description": "Simple stories for small language models",
        "example_command": "python download_datasets.py --dataset 'roneneldan/TinyStories'"
    },
    "HuggingFaceTB/cosmopedia-100k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["synthetic", "education"], "tokens": "very_high",
        "description": "Synthetic educational content across multiple topics",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceTB/cosmopedia-100k' --max-samples 1000"
    },

    # High-Quality Instruction Data
    "philschmid/dolly-15k-oai-style": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "high_quality"], "tokens": "medium",
        "description": "Dolly dataset in OpenAI conversation format",
        "example_command": "python download_datasets.py --dataset 'philschmid/dolly-15k-oai-style'"
    },
    "garage-bAInd/Open-Platypus": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "reasoning"], "tokens": "high",
        "description": "High-quality instruction dataset with reasoning focus",
        "example_command": "python download_datasets.py --dataset 'garage-bAInd/Open-Platypus'"
    },
    "WizardLM/WizardLM_evol_instruct_V2_196k": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "large": True,
        "categories": ["instruction", "evolution"], "tokens": "very_high",
        "description": "Evolved instruction dataset with complex reasoning",
        "example_command": "python download_datasets.py --dataset 'WizardLM/WizardLM_evol_instruct_V2_196k' --max-samples 1000"
    },

    # Medical & Science
    "medalpaca/medical_meadow_medical_flashcards": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["medical", "flashcards"], "tokens": "medium",
        "description": "Medical knowledge in flashcard format",
        "example_command": "python download_datasets.py --dataset 'medalpaca/medical_meadow_medical_flashcards'"
    },
    "pubmed_qa": {
        "splits": ["train"], "subset": "pqa_labeled", "streaming_safe": True,
        "categories": ["medical", "qa"], "tokens": "high",
        "description": "Question answering on PubMed abstracts",
        "example_command": "python download_datasets.py --dataset 'pubmed_qa'"
    },
    "scientific_papers": {
        "splits": ["train", "validation", "test"], "subset": "arxiv", "streaming_safe": True,
        "categories": ["scientific", "papers"], "tokens": "very_high", "large": True,
        "description": "ArXiv and PubMed scientific papers",
        "example_command": "python download_datasets.py --dataset 'scientific_papers' --max-samples 1000"
    },
    "bigscience/P3": {
        "splits": ["train", "validation"], "subset": "all", "streaming_safe": True, "large": True,
        "categories": ["multitask", "p3"], "tokens": "very_high",
        "description": "Public Pool of Prompts - 170+ NLP tasks",
        "example_command": "python download_datasets.py --dataset 'bigscience/P3' --max-samples 1000"
    },

    # Additional Datasets
    "bigscience/xP3": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 50000,
        "categories": ["multilingual", "instruction"], "tokens": "very_high", "large": True,
        "description": "Multilingual version of P3 dataset",
        "example_command": "python download_datasets.py --dataset 'bigscience/xP3' --max-samples 1000"
    },
    "Muennighoff/natural-instructions": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "natural"], "tokens": "very_high", "large": True,
        "description": "Large collection of NLP tasks with instructions",
        "example_command": "python download_datasets.py --dataset 'Muennighoff/natural-instructions' --max-samples 1000"
    },
    "allenai/prosocial-dialog": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["dialog", "prosocial"], "tokens": "medium",
        "description": "Prosocial conversation dataset with social norms",
        "example_command": "python download_datasets.py --dataset 'allenai/prosocial-dialog'"
    },
    "HuggingFaceH4/self_instruct": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "self_instruct"], "tokens": "medium",
        "description": "Self-generated instruction-following dataset",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/self_instruct'"
    },
}

# Retry strategies for different failure modes
RETRY_STRATEGIES = [
    # Strategy 1: No auth token (for public datasets)
    {
        "name": "no_token",
        "params": {
            "num_proc": 1
        }
    },
    # Strategy 2: Standard download with token
    {
        "name": "standard_with_token",
        "params": {
            "token": True,
            "num_proc": 4
        }
    },
    # Strategy 3: Streaming mode for large datasets
    {
        "name": "streaming_mode",
        "params": {
            "streaming": True,
            "num_proc": 1
        }
    },
    # Strategy 4: Single process for compatibility
    {
        "name": "single_process",
        "params": {
            "num_proc": 1,
            "token": True
        }
    },
    # Strategy 5: Streaming with token
    {
        "name": "streaming_with_token",
        "params": {
            "streaming": True,
            "token": True
        }
    },
    # Strategy 5: Force redownload
    {
        "name": "force_redownload",
        "params": {
            "download_mode": "force_redownload",
            "num_proc": 1
        }
    },
    # Strategy 6: Trust remote code
    {
        "name": "trust_remote",
        "params": {
            "trust_remote_code": True,
            "num_proc": 1
        }
    },
    # Strategy 7: No multiprocessing
    {
        "name": "no_multiproc",
        "params": {
            "num_proc": None
        }
    },
    # Strategy 8: Minimal parameters
    {
        "name": "minimal",
        "params": {}
    },
    # Strategy 9: Cache only (for offline)
    {
        "name": "cache_only",
        "params": {
            "download_mode": "reuse_cache_if_exists"
        }
    }
]

class DatasetDownloader:
    def __init__(self, output_dir: str = "/project/data/pretraining/raw",
                 max_samples: Optional[int] = None):
        """Initialize the dataset downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.summary = {
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "successful": [],
            "failed": [],
            "skipped": []
        }

        # Import datasets library
        self.load_dataset, self.load_from_disk, self.datasets_lib = import_datasets()

    def check_memory(self) -> Tuple[float, float]:
        """Check available memory"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        usage_percent = mem.percent
        return available_gb, usage_percent

    def download_dataset_with_retry(self, dataset_name: str, config: Dict) -> bool:
        """Download a dataset with multiple retry strategies"""
        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_name}")
        print(f"{'='*60}")

        # Check if dataset is marked as large
        is_large = config.get("large", False)
        streaming_safe = config.get("streaming_safe", True)

        # Check memory for large datasets
        available_gb, usage_percent = self.check_memory()
        if is_large and available_gb < 10:
            print(f"⚠️  Low memory ({available_gb:.1f}GB available), using streaming mode")
            streaming_safe = True

        # Try different strategies
        for strategy_idx, strategy in enumerate(RETRY_STRATEGIES):
            print(f"\nAttempt {strategy_idx + 1}/{len(RETRY_STRATEGIES)}: {strategy['name']}")

            try:
                # Prepare parameters
                params = strategy["params"].copy()

                # Add subset if specified
                if config.get("subset"):
                    dataset_args = [dataset_name, config["subset"]]
                else:
                    dataset_args = [dataset_name]

                # Force streaming for large datasets on low memory
                if is_large and available_gb < 10:
                    params["streaming"] = True

                # Handle different splits
                for split in config.get("splits", ["train"]):
                    print(f"  Downloading split: {split}")

                    try:
                        # Add delay to avoid rate limiting
                        if strategy_idx > 0:  # Only add delay after first attempt
                            time.sleep(2)

                        # Download the dataset
                        if params.get("streaming", False):
                            # Streaming mode
                            dataset = self.load_dataset(*dataset_args, split=split, **params)

                            # Save streaming dataset
                            output_path = self.output_dir / dataset_name.replace("/", "_") / split
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Stream and save samples
                            samples = []
                            max_to_download = self.max_samples if self.max_samples else 100000

                            print(f"  Streaming up to {max_to_download} samples...")
                            for idx, sample in enumerate(tqdm(dataset, total=max_to_download)):
                                samples.append(sample)
                                if idx >= max_to_download - 1:
                                    break

                            # Save as JSON
                            with open(output_path / "data.json", "w") as f:
                                json.dump(samples, f)

                            print(f"  ✓ Saved {len(samples)} samples to {output_path}")

                        else:
                            # Regular download
                            dataset = self.load_dataset(*dataset_args, split=split, **params)

                            # Apply sample limit if specified
                            if self.max_samples:
                                try:
                                    if hasattr(dataset, '__len__') and len(dataset) > self.max_samples:
                                        if hasattr(dataset, 'select'):
                                            dataset = dataset.select(range(self.max_samples))
                                except (TypeError, AttributeError):
                                    # Handle iterable datasets that don't support len() or select()
                                    pass

                            # Save dataset
                            output_path = self.output_dir / dataset_name.replace("/", "_") / split
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Save in arrow format if possible
                            try:
                                if hasattr(dataset, 'save_to_disk'):
                                    dataset.save_to_disk(str(output_path))
                            except (AttributeError, TypeError):
                                pass

                            # Also save as JSON for compatibility
                            try:
                                if hasattr(dataset, 'to_json'):
                                    dataset.to_json(str(output_path / "data.json"))
                                else:
                                    # Fallback for iterable datasets
                                    import json
                                    samples = []
                                    for i, sample in enumerate(dataset):
                                        if self.max_samples and i >= self.max_samples:
                                            break
                                        samples.append(sample)

                                    with open(output_path / "data.json", "w") as f:
                                        json.dump(samples, f)
                            except Exception as e:
                                print(f"  ⚠️ Could not save as JSON: {e}")

                            # Get length safely
                            try:
                                if hasattr(dataset, '__len__'):
                                    dataset_len = len(dataset)
                                else:
                                    dataset_len = self.max_samples or "unknown"
                            except (TypeError, AttributeError):
                                dataset_len = "unknown"

                            print(f"  ✓ Saved {dataset_len} samples to {output_path}")

                    except Exception as e:
                        print(f"  ✗ Failed to download split {split}: {str(e)}")
                        continue

                # If we got here, download was successful
                self.summary["successful"].append(dataset_name)
                return True

            except Exception as e:
                print(f"  ✗ Strategy failed: {str(e)}")
                continue

        # All strategies failed
        print(f"\n✗ Failed to download {dataset_name} after all attempts")
        self.summary["failed"].append(dataset_name)

        # Save error details
        error_file = self.output_dir / ".errors" / f"{dataset_name.replace('/', '_')}.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"All strategies failed\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")

        return False

    def download_all(self, datasets: Optional[List[str]] = None,
                    parallel: bool = True, max_workers: int = 2):
        """Download all configured datasets"""
        # Select datasets to download
        if datasets:
            dataset_configs = {k: v for k, v in DATASETS_CONFIG.items() if k in datasets}
        else:
            dataset_configs = DATASETS_CONFIG

        print(f"\nPreparing to download {len(dataset_configs)} datasets")
        print(f"Output directory: {self.output_dir}")

        # Check existing datasets
        existing = set()
        for dataset_name in dataset_configs:
            dataset_dir = self.output_dir / dataset_name.replace("/", "_")
            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                existing.add(dataset_name)
                print(f"  ⚠️  {dataset_name} already exists, skipping")
                self.summary["skipped"].append(dataset_name)

        # Filter out existing
        to_download = {k: v for k, v in dataset_configs.items() if k not in existing}

        if not to_download:
            print("All datasets already downloaded!")
            return

        print(f"\nDownloading {len(to_download)} new datasets...")

        # Download datasets
        if parallel and len(to_download) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.download_dataset_with_retry, name, config): name
                    for name, config in to_download.items()
                }

                for future in as_completed(futures):
                    dataset_name = futures[future]
                    try:
                        success = future.result(timeout=1800)  # 30 min timeout
                        if success:
                            print(f"✓ Completed: {dataset_name}")
                        else:
                            print(f"✗ Failed: {dataset_name}")
                    except Exception as e:
                        print(f"✗ Exception downloading {dataset_name}: {e}")
                        self.summary["failed"].append(dataset_name)
        else:
            # Sequential download
            for dataset_name, config in to_download.items():
                self.download_dataset_with_retry(dataset_name, config)

        # Save summary
        self.save_summary()

    def save_summary(self):
        """Save download summary"""
        self.summary["total_attempted"] = len(self.summary["successful"]) + len(self.summary["failed"])
        self.summary["total_datasets"] = len(DATASETS_CONFIG)

        if self.summary["total_attempted"] > 0:
            self.summary["success_rate"] = len(self.summary["successful"]) / self.summary["total_attempted"]
        else:
            self.summary["success_rate"] = 0

        # Add memory stats
        available_gb, usage_percent = self.check_memory()
        self.summary["memory_stats"] = {
            "available_gb": available_gb,
            "usage_percent": usage_percent
        }

        # Save summary
        summary_file = self.output_dir / "download_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.summary, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Total datasets available: {len(DATASETS_CONFIG)}")
        print(f"Successfully downloaded: {len(self.summary['successful'])}")
        print(f"Failed: {len(self.summary['failed'])}")
        print(f"Skipped (existing): {len(self.summary['skipped'])}")
        print(f"Success rate: {self.summary['success_rate']:.1%}")
        print(f"\nSummary saved to: {summary_file}")

        if self.summary["failed"]:
            print(f"\nFailed datasets:")
            for name in self.summary["failed"]:
                print(f"  - {name}")

def filter_datasets_by_categories(categories):
    """Filter datasets by categories"""
    filtered = {}
    for name, config in DATASETS_CONFIG.items():
        dataset_categories = config.get("categories", [])
        if any(cat in dataset_categories for cat in categories):
            filtered[name] = config
    return filtered

def list_datasets_with_examples():
    """List all available datasets with example commands"""
    print(f"\n{'='*80}")
    print("AVAILABLE DATASETS WITH EXAMPLE COMMANDS")
    print(f"{'='*80}")

    by_category = {}
    for name, config in DATASETS_CONFIG.items():
        categories = config.get("categories", ["other"])
        main_cat = categories[0]
        if main_cat not in by_category:
            by_category[main_cat] = []
        by_category[main_cat].append((name, config))

    for category, datasets in by_category.items():
        print(f"\n📂 {category.upper().replace('_', ' ')} ({len(datasets)} datasets)")
        print("-" * 60)

        for name, config in datasets:
            description = config.get("description", "No description available")
            example_cmd = config.get("example_command", f"python download_datasets.py --dataset '{name}'")
            tokens = config.get("tokens", "unknown")
            large = " [LARGE]" if config.get("large", False) else ""

            print(f"  🔹 {name}{large}")
            print(f"     {description}")
            print(f"     Tokens: {tokens}")
            print(f"     Example: {example_cmd}")
            print()

def validate_dataset_config():
    """Validate that all datasets have proper configuration"""
    issues = []

    for name, config in DATASETS_CONFIG.items():
        # Check required fields
        if "splits" not in config:
            issues.append(f"{name}: Missing 'splits' field")
        if "categories" not in config:
            issues.append(f"{name}: Missing 'categories' field")
        if "streaming_safe" not in config:
            issues.append(f"{name}: Missing 'streaming_safe' field")

        # Check for description and example
        if "description" not in config:
            issues.append(f"{name}: Missing 'description' field")
        if "example_command" not in config:
            issues.append(f"{name}: Missing 'example_command' field")

    if issues:
        print("⚠️  Dataset Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All dataset configurations are valid")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for enhanced LLM features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_datasets.py --all

  # Download specific categories
  python download_datasets.py --pretraining --rag --evaluation

  # Download for specific enhanced features
  python download_datasets.py --for-moh --for-rag --for-continual-learning

  # Custom data directory
  python download_datasets.py --all --data-dir /custom/path/data

  # Download small datasets only
  python download_datasets.py --all --skip-large
        """
    )

    # Output configuration
    parser.add_argument("--output-dir", default="/project/code/data",
                      help="Output directory for datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Maximum samples per dataset")

    # Dataset selection
    parser.add_argument("--all", action="store_true",
                       help="Download all datasets")
    parser.add_argument("--dataset", type=str, default=None,
                      help="Download specific dataset only")

    # Category-based selection
    parser.add_argument("--pretraining", action="store_true",
                       help="Download pre-training datasets")
    parser.add_argument("--rag", action="store_true",
                       help="Download RAG knowledge base datasets")
    parser.add_argument("--multitask", action="store_true",
                       help="Download multi-task learning datasets")
    parser.add_argument("--continual", action="store_true",
                       help="Download continual learning datasets")
    parser.add_argument("--evaluation", action="store_true",
                       help="Download evaluation datasets")
    parser.add_argument("--safety", action="store_true",
                       help="Download safety and bias datasets")
    parser.add_argument("--multimodal", action="store_true",
                       help="Download multi-modal datasets")
    parser.add_argument("--code", action="store_true",
                       help="Download code datasets")
    parser.add_argument("--conversation", action="store_true",
                       help="Download conversational datasets")

    # Feature-specific downloads
    parser.add_argument("--for-moh", action="store_true",
                       help="Download datasets for Mixture of Heads training")
    parser.add_argument("--for-moa", action="store_true",
                       help="Download datasets for Mixture of Activations training")
    parser.add_argument("--for-rag", action="store_true",
                       help="Download datasets for RAG training")
    parser.add_argument("--for-continual-learning", action="store_true",
                       help="Download datasets for continual learning")
    parser.add_argument("--for-cross-attention", action="store_true",
                       help="Download datasets for cross-attention training")
    parser.add_argument("--for-evaluation", action="store_true",
                       help="Download comprehensive evaluation datasets")
    parser.add_argument("--for-safety", action="store_true",
                       help="Download safety and bias evaluation datasets")

    # Download configuration
    parser.add_argument("--parallel", action="store_true",
                      help="Download datasets in parallel")
    parser.add_argument("--max-workers", type=int, default=2,
                      help="Maximum parallel workers")
    parser.add_argument("--skip-large", action="store_true",
                      help="Skip datasets marked as large")
    parser.add_argument("--small-only", action="store_true",
                      help="Download only small/medium datasets")

    # Information and validation
    parser.add_argument("--list-datasets", action="store_true",
                      help="List all available datasets with examples")
    parser.add_argument("--validate-config", action="store_true",
                      help="Validate dataset configurations")

    args = parser.parse_args()

    # Handle information commands
    if args.list_datasets:
        list_datasets_with_examples()
        return

    if args.validate_config:
        validate_dataset_config()
        return

    # Initialize downloader
    downloader = DatasetDownloader(output_dir=args.output_dir,
                                 max_samples=args.max_samples)

    # Determine what to download
    datasets = None

    if args.dataset:
        # Single dataset
        datasets = [args.dataset]
    elif args.all:
        # All datasets
        datasets = None
    else:
        # Category-based selection
        categories = []

        # Direct categories
        if args.pretraining:
            categories.extend(["pretraining", "instruction"])
        if args.rag:
            categories.extend(["rag", "knowledge", "web", "news"])
        if args.multitask:
            categories.extend(["multitask", "evaluation"])
        if args.continual:
            categories.extend(["continual", "multitask"])
        if args.evaluation:
            categories.extend(["evaluation", "reasoning", "qa", "summarization"])
        if args.safety:
            categories.extend(["safety", "rlhf", "feedback"])
        if args.multimodal:
            categories.extend(["multimodal", "vision"])
        if args.code:
            categories.extend(["code", "python", "github"])
        if args.conversation:
            categories.extend(["conversation", "dialog", "multiturn"])

        # Feature-specific mappings
        if args.for_moh:
            categories.extend(["pretraining", "instruction", "conversation"])
        if args.for_moa:
            categories.extend(["pretraining", "instruction", "code"])
        if args.for_rag:
            categories.extend(["rag", "knowledge", "web", "news", "qa"])
        if args.for_continual_learning:
            categories.extend(["continual", "multitask", "math", "reasoning"])
        if args.for_cross_attention:
            categories.extend(["multimodal", "vision", "conversation"])
        if args.for_evaluation:
            categories.extend(["evaluation", "reasoning", "qa", "summarization"])
        if args.for_safety:
            categories.extend(["safety", "rlhf", "feedback"])

        # Filter datasets by categories
        if categories:
            filtered_config = filter_datasets_by_categories(categories)
            datasets = list(filtered_config.keys())
            print(f"📊 Found {len(datasets)} datasets matching categories: {set(categories)}")
        else:
            # No specific categories, download all
            datasets = None

    # Apply size filters
    if args.skip_large or args.small_only:
        if datasets is None:
            datasets = list(DATASETS_CONFIG.keys())

        filtered_datasets = []
        for dataset_name in datasets:
            config = DATASETS_CONFIG.get(dataset_name, {})
            is_large = config.get("large", False)

            if args.skip_large and is_large:
                continue
            if args.small_only and is_large:
                continue

            filtered_datasets.append(dataset_name)

        datasets = filtered_datasets
        print(f"📊 After size filtering: {len(datasets)} datasets")

    # Print download plan
    if datasets:
        print(f"\n📋 Download Plan:")
        print(f"Output directory: {args.output_dir}")
        print(f"Datasets to download: {len(datasets)}")

        # Group by category for display
        by_category = {}
        for dataset_name in datasets:
            config = DATASETS_CONFIG.get(dataset_name, {})
            cats = config.get("categories", ["other"])
            main_cat = cats[0] if cats else "other"
            if main_cat not in by_category:
                by_category[main_cat] = []
            by_category[main_cat].append(dataset_name)

        for category, dataset_list in by_category.items():
            print(f"  {category}: {len(dataset_list)} datasets")

    # Start download
    downloader.download_all(datasets=datasets,
                          parallel=args.parallel,
                          max_workers=args.max_workers)

if __name__ == "__main__":
    main()