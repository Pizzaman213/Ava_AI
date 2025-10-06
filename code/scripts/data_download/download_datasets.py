#!/usr/bin/env python3
"""
Comprehensive Dataset Downloader for Enhanced LLM Features

DEFAULT BEHAVIOR: Downloads ALL available datasets when run without arguments.
This ensures comprehensive training data covering all domains and capabilities.

This script downloads and prepares datasets for all enhanced features including:
- ANTHROPIC DATASETS (HH-RLHF, model-written-evals, persona, sycophancy, AI risk)
- Constitutional AI & RLAIF datasets
- Pre-training datasets (OpenWebText, The Pile, WikiText, BookCorpus)
- RAG knowledge bases (Wikipedia, MS MARCO, Natural Questions)
- Multi-task learning datasets (GLUE, SuperGLUE, XTREME)
- Evaluation datasets (HellaSwag, ARC, MMLU, CNN/DailyMail)
- Safety & bias datasets (Toxicity, bias evaluation, red-teaming)
- Preference learning datasets (RLHF, DPO, feedback)
- Multi-modal datasets (Vision-language, audio-text)
- Continual learning datasets (For episodic memory)
- Code datasets (For programming capabilities)

Supports 80+ diverse datasets with multiple retry strategies and feature-specific groupings.
All Anthropic datasets are downloaded by default to ensure safety and alignment training.
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
from typing import Dict, List, Optional, Tuple
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
    # Core Instruction Tuning (Verified Working) - HIGHEST QUALITY
    "teknium/OpenHermes-2.5": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction", "qa"], "tokens": "high",
        "quality_score": 10,  # GPT-4 quality
        "priority": 1,  # Download first
        "description": "1M+ GPT-4 generated Q&A pairs - highest quality instruction dataset",
        "example_command": "python download_datasets.py --dataset 'teknium/OpenHermes-2.5'"
    },
    "Open-Orca/OpenOrca": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction", "qa"], "tokens": "very_high",
        "quality_score": 9.5,  # GPT-4/3.5 quality
        "priority": 1,  # Download early
        "description": "4M GPT-4/GPT-3.5 instruction Q&A pairs from FLAN",
        "example_command": "python download_datasets.py --dataset 'Open-Orca/OpenOrca'"
    },
    "meta-math/MetaMathQA": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "math", "qa"], "tokens": "high",
        "quality_score": 9,  # High-quality math
        "priority": 2,  # Important for reasoning
        "description": "395k mathematical Q&A with step-by-step solutions",
        "example_command": "python download_datasets.py --dataset 'meta-math/MetaMathQA'"
    },
    "m-a-p/Code-Feedback": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "code", "qa"], "tokens": "high",
        "quality_score": 8,  # Good code Q&A
        "priority": 3,
        "description": "Code Q&A dataset with detailed feedback",
        "example_command": "python download_datasets.py --dataset 'm-a-p/Code-Feedback'"
    },

    # OpenAssistant (Verified Working)
    "OpenAssistant/oasst1": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "conversation", "qa"], "tokens": "high",
        "quality_score": 8,  # Human quality
        "description": "Human-generated, assistant-ranked conversation Q&A trees",
        "example_command": "python download_datasets.py --dataset 'OpenAssistant/oasst1'"
    },
    "OpenAssistant/oasst2": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "conversation", "qa"], "tokens": "high",
        "quality_score": 8.5,  # Improved human quality
        "priority": 2,
        "description": "Enhanced human-ranked conversational Q&A dataset",
        "example_command": "python download_datasets.py --dataset 'OpenAssistant/oasst2'"
    },

    # ================================
    # RAG KNOWLEDGE BASES
    # ================================
    # Large-scale Text Datasets (High Token Count)
    ###
    "allenai/c4": {
        "splits": ["train"], "subset": "en", "streaming_safe": True, "max_samples": 100000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "Colossal Clean Crawled Corpus - cleaned web text for language modeling",
        "example_command": "python download_datasets.py --dataset 'allenai/c4' --max-samples 10000"
    },
    
    "openwebtext": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 500000,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "description": "Open-source recreation of GPT-2's WebText training dataset",
        "example_command": "python download_datasets.py --dataset 'openwebtext' --max-samples 5000"
    },
    "EleutherAI/pile": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 5000000,
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
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 750000,
        "categories": ["rag", "news"], "tokens": "very_high", "large": True,
        "description": "News articles from Common Crawl for current events knowledge",
        "example_command": "python download_datasets.py --dataset 'cc_news' --max-samples 5000"
    },
    # RedPajama dataset removed - requires special handling with subsets
    # Use alternative datasets like c4, openwebtext, or fineweb instead
    "HuggingFaceFW/fineweb": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 2000000,
        "categories": ["rag", "web"], "tokens": "very_high", "large": True,
        "description": "High-quality web text filtered from CommonCrawl",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceFW/fineweb' --max-samples 5000"
    },
    "HuggingFaceFW/fineweb-edu": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 500000,
        "categories": ["rag", "education"], "tokens": "very_high", "large": True,
        "description": "Educational web content from FineWeb corpus",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceFW/fineweb-edu' --max-samples 5000"
    },
    "tiiuae/falcon-refinedweb": {
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 300000,
        "categories": ["rag", "web"], "tokens": "very_high", "large": True,
        "description": "Refined web text used to train Falcon LLM",
        "example_command": "python download_datasets.py --dataset 'tiiuae/falcon-refinedweb' --max-samples 3000"
    },

    # ================================
    # MULTI-TASK & CONTINUAL LEARNING
    # ================================
    # Math & Reasoning (For continual learning)
    "openai/gsm8k": {
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
    "nyu-mll/glue": {
        "splits": ["train", "validation"], "subset": "cola", "streaming_safe": True,
        "categories": ["multitask", "evaluation"], "tokens": "low",
        "task_type": "classification",
        "description": "GLUE CoLA task - linguistic acceptability classification",
        "example_command": "python download_datasets.py --dataset 'glue'"
    },
    "rajpurkar/squad": {
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
        "splits": ["train"], "subset": None, "streaming_safe": True, "max_samples": 400000,
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
    "allenai/winogrande": {
        "splits": ["train", "validation"], "subset": "winogrande_xl", "streaming_safe": True,
        "categories": ["evaluation", "reasoning"], "tokens": "low",
        "description": "Commonsense reasoning with pronoun resolution",
        "example_command": "python download_datasets.py --dataset 'winogrande'"
    },
    "Rowan/hellaswag": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "commonsense"], "tokens": "medium",
        "description": "Commonsense natural language inference",
        "example_command": "python download_datasets.py --dataset 'hellaswag'"
    },
    "rajpurkar/squad": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "qa"], "tokens": "medium",
        "description": "Stanford Question Answering Dataset for reading comprehension",
        "example_command": "python download_datasets.py --dataset 'squad'"
    },
    "rajpurkar/squad_v2": {
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
    "abisee/cnn_dailymail": {
        "splits": ["train", "validation", "test"], "subset": "3.0.0", "streaming_safe": True,
        "categories": ["evaluation", "summarization"], "tokens": "high",
        "description": "CNN/DailyMail news articles with highlights for summarization",
        "example_command": "python download_datasets.py --dataset 'cnn_dailymail'"
    },

    # ================================
    # ANTHROPIC DATASETS (COMPLETE COLLECTION)
    # ================================
    "Anthropic/hh-rlhf": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "safety", "rlhf", "preference", "harmlessness", "helpfulness"],
        "tokens": "high",
        "quality_score": 9,  # Anthropic's high-quality human feedback
        "priority": 1,  # Critical for safety training
        "description": "Anthropic's human preference data for helpful and harmless AI assistant training",
        "example_command": "python download_datasets.py --dataset 'Anthropic/hh-rlhf'"
    },
    "Anthropic/model-written-evals": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "evaluation", "persona", "sycophancy", "ai_risk", "bias"],
        "tokens": "medium",
        "quality_score": 9,  # High-quality AI-generated evals
        "priority": 1,  # Important for model evaluation
        "description": "Anthropic's model-written evaluations for persona, sycophancy, AI risks, and gender bias",
        "example_command": "python download_datasets.py --dataset 'Anthropic/model-written-evals'"
    },
    "HyperionHF/Anthropic-evals-persona": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "persona", "evaluation", "behavior"],
        "tokens": "medium",
        "quality_score": 8,  # Derived from Anthropic evals
        "priority": 2,
        "description": "Persona evaluation dataset based on Anthropic's model-written evaluations",
        "example_command": "python download_datasets.py --dataset 'HyperionHF/Anthropic-evals-persona'"
    },
    "HuggingFaceH4/helpful-anthropic-raw": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "helpfulness", "raw_data", "conversation"],
        "tokens": "high",
        "quality_score": 8,  # Raw Anthropic data
        "priority": 2,
        "description": "Raw helpful conversations from Anthropic's research",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/helpful-anthropic-raw'"
    },
    "Baidicoot/anthropic-harmless-rlhf": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "harmlessness", "rlhf", "safety"],
        "tokens": "medium",
        "quality_score": 8,  # Derived from Anthropic
        "priority": 2,
        "description": "Harmlessness-focused subset of Anthropic's RLHF data",
        "example_command": "python download_datasets.py --dataset 'Baidicoot/anthropic-harmless-rlhf'"
    },
    # Trelis/hh-rlhf-dpo removed - requires authentication (gated dataset)
    # Alternative: Use the original Anthropic/hh-rlhf which is already included

    # ================================
    # CONSTITUTIONAL AI & RLAIF DATASETS
    # ================================
    # Note: Some Constitutional AI datasets may have different split names
    "HuggingFaceH4/cai-conversation-harmless": {
        "splits": ["train_sft", "test_sft"], "subset": None, "streaming_safe": True,
        "categories": ["constitutional_ai", "rlaif", "harmlessness", "synthetic"],
        "tokens": "high",
        "quality_score": 8,  # AI feedback quality
        "priority": 2,
        "description": "Constitutional AI conversations focused on harmlessness",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/cai-conversation-harmless'"
    },

    # ================================
    # SAFETY & BIAS DATASETS (EXTENDED)
    # ================================
    "HuggingFaceH4/ultrafeedback_binarized": {
        "splits": ["train_prefs", "test_prefs"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "feedback", "preference", "rlhf"], "tokens": "high",
        "quality_score": 8.5,  # High-quality preferences
        "priority": 2,
        "description": "High-quality preference data for RLHF from GPT-4 feedback",
        "example_command": "python download_datasets.py --dataset 'HuggingFaceH4/ultrafeedback_binarized'"
    },
    "PKU-Alignment/PKU-SafeRLHF": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "rlhf", "harmlessness", "red_teaming"],
        "tokens": "high",
        "quality_score": 8,  # Safety-focused
        "priority": 2,
        "description": "Safety-focused RLHF dataset from PKU with red-teaming examples",
        "example_command": "python download_datasets.py --dataset 'PKU-Alignment/PKU-SafeRLHF'"
    },
    "allenai/real-toxicity-prompts": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "toxicity", "red_teaming", "bias"],
        "tokens": "medium",
        "quality_score": 7.5,  # Toxicity detection
        "priority": 3,
        "description": "Real toxicity prompts for testing model safety",
        "example_command": "python download_datasets.py --dataset 'allenai/real-toxicity-prompts'"
    },
    "google/civil_comments": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "bias", "toxicity", "fairness"],
        "tokens": "high",
        "quality_score": 8,  # High-quality toxicity annotations
        "priority": 2,
        "description": "Civil Comments dataset with toxicity and identity annotations",
        "example_command": "python download_datasets.py --dataset 'google/civil_comments'"
    },
    "SetFit/toxic_conversations": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "toxicity", "conversation"],
        "tokens": "medium",
        "quality_score": 7.5,  # Jigsaw toxicity data
        "priority": 3,
        "description": "Toxic conversations from Jigsaw Unintended Bias challenge",
        "example_command": "python download_datasets.py --dataset 'SetFit/toxic_conversations'"
    },

    # ================================
    # MULTI-MODAL DATASETS
    # ================================
    # ShareGPT4V removed - access issues

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
    # lmsys-chat-1m removed - access issues
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
    # scientific_papers removed - access issues
    # bigscience/P3 removed - access issues

    # Additional Datasets
    # bigscience/xP3 removed - access issues
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
    # Strategy 1: Streaming mode (primary strategy)
    {
        "name": "streaming_mode",
        "params": {
            "streaming": True,
            "cache_dir": None,
            "download_mode": "force_redownload"
        }
    },
    # Strategy 2: Streaming with token
    {
        "name": "streaming_with_token",
        "params": {
            "streaming": True,
            "token": True,
            "cache_dir": None,
            "download_mode": "force_redownload"
        }
    },
    # Strategy 3: No cache standard download
    {
        "name": "no_cache_download",
        "params": {
            "cache_dir": None,
            "download_mode": "force_redownload",
            "num_proc": 1
        }
    },
    # Strategy 4: No cache with token
    {
        "name": "no_cache_with_token",
        "params": {
            "token": True,
            "cache_dir": None,
            "download_mode": "force_redownload",
            "num_proc": 1
        }
    },
    # Strategy 5: Streaming trust remote
    {
        "name": "streaming_trust_remote",
        "params": {
            "streaming": True,
            "trust_remote_code": True,
            "cache_dir": None,
            "download_mode": "force_redownload"
        }
    },
    # Strategy 6: Trust remote no cache
    {
        "name": "trust_remote_no_cache",
        "params": {
            "trust_remote_code": True,
            "cache_dir": None,
            "download_mode": "force_redownload",
            "num_proc": 1
        }
    },
    # Strategy 7: Minimal streaming
    {
        "name": "minimal_streaming",
        "params": {
            "streaming": True
        }
    },
    # Strategy 8: Fallback no cache
    {
        "name": "fallback_no_cache",
        "params": {
            "cache_dir": None
        }
    }
]

class DatasetDownloader:
    def __init__(self, output_dir: str = "/project/code/data",
                 max_samples: Optional[int] = None,
                 batch_size: int = 1000):
        """Initialize the dataset downloader with memory-efficient batching by default"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.summary = {
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "successful": [],
            "failed": [],
            "skipped": []
        }

        # Import datasets library
        self.load_dataset, self.load_from_disk, self.datasets_lib = import_datasets()

        # Try to login to HuggingFace if token exists
        try:
            from huggingface_hub import login
            # Check for token in environment or default location
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                login(token=token)
                print("✓ Logged in to HuggingFace")
        except:
            pass

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

        # Check memory for large datasets
        available_gb, _ = self.check_memory()
        if is_large and available_gb < 10:
            print(f"⚠️  Low memory ({available_gb:.1f}GB available), using streaming mode")

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

                # Always use streaming and no cache
                params["streaming"] = True
                params["cache_dir"] = None
                if "download_mode" not in params:
                    params["download_mode"] = "force_redownload"

                # Handle different splits
                for split in config.get("splits", ["train"]):
                    print(f"  Downloading split: {split}")

                    try:
                        # Add delay to avoid rate limiting
                        if strategy_idx > 0:  # Only add delay after first attempt
                            time.sleep(2)

                        # Download the dataset (always streaming)
                        # Streaming mode
                        # Fix: Use try/except and handle authentication
                        dataset = None
                        try:
                            dataset = self.load_dataset(*dataset_args, split=split, **params)
                        except Exception as load_error:
                            # Try without any extra params as fallback
                            if "LocalEntryNotFoundError" in str(load_error) or "Couldn't find" in str(load_error):
                                print(f"  Retrying with basic parameters...")
                                try:
                                    dataset = self.load_dataset(*dataset_args, split=split, streaming=True)
                                except:
                                    print(f"  ✗ Could not load dataset even with basic params")
                                    continue

                        if dataset is None:
                            print(f"  ✗ Failed to load dataset")
                            continue

                        # Save streaming dataset to raw folder
                        output_path = self.output_dir / "raw" / dataset_name.replace("/", "_") / split
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Stream and save samples in batches to avoid memory issues
                        batch_size = self.batch_size  # Use configurable batch size
                        batch = []
                        batch_num = 0
                        total_saved = 0
                        max_to_download = self.max_samples if self.max_samples else 100000

                        print(f"  Streaming up to {max_to_download} samples in batches of {batch_size}...")

                        with tqdm(total=max_to_download) as pbar:
                            for idx, sample in enumerate(dataset):
                                batch.append(sample)
                                pbar.update(1)

                                # Save batch when it reaches batch_size or we hit the limit
                                if len(batch) >= batch_size or idx >= max_to_download - 1:
                                    # Save batch as separate JSON file
                                    batch_file = output_path / f"batch_{batch_num:04d}.json"
                                    with open(batch_file, "w") as f:
                                        json.dump(batch, f)

                                    total_saved += len(batch)
                                    print(f"    Saved batch {batch_num}: {len(batch)} samples ({total_saved} total)")

                                    # Clear batch and check memory
                                    batch = []
                                    batch_num += 1

                                    # Memory check every 10 batches
                                    if batch_num % 10 == 0:
                                        available_gb, usage_percent = self.check_memory()
                                        if usage_percent > 85:
                                            print(f"    ⚠️  High memory usage ({usage_percent:.1f}%), pausing briefly...")
                                            time.sleep(1)

                                if idx >= max_to_download - 1:
                                    break

                        # Create summary file for this dataset
                        summary = {
                            "total_samples": total_saved,
                            "num_batches": batch_num,
                            "batch_size": batch_size,
                            "dataset_name": dataset_name,
                            "split": split
                        }
                        with open(output_path / "summary.json", "w") as f:
                            json.dump(summary, f, indent=2)

                        print(f"  ✓ Saved {total_saved} samples in {batch_num} batches to {output_path}")

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
        error_file = self.output_dir / "raw" / ".errors" / f"{dataset_name.replace('/', '_')}.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"All strategies failed\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")

        return False

    def download_all(self, datasets: Optional[List[str]] = None,
                    parallel: bool = True, max_workers: int = 4):
        """Download all configured datasets"""
        # Select datasets to download
        if datasets:
            dataset_configs = {k: v for k, v in DATASETS_CONFIG.items() if k in datasets}
        else:
            dataset_configs = DATASETS_CONFIG

        print(f"\nPreparing to download {len(dataset_configs)} datasets")
        print(f"Output directory: {self.output_dir}")

        # Check existing datasets in raw folder
        existing = set()
        for dataset_name in dataset_configs:
            dataset_dir = self.output_dir / "raw" / dataset_name.replace("/", "_")
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
        description="Download datasets for enhanced LLM features (DEFAULT: Downloads ALL datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DEFAULT: Download ALL datasets (no arguments needed)
  python download_datasets.py

  # Download all datasets explicitly (same as default)
  python download_datasets.py --all

  # Download only core Anthropic datasets
  python download_datasets.py --anthropic

  # Download specific categories
  python download_datasets.py --pretraining --rag --evaluation

  # Download safety & alignment datasets (Constitutional AI, RLHF, etc.)
  python download_datasets.py --safety --constitutional-ai --harmlessness

  # Download preference learning datasets (RLHF, DPO, feedback)
  python download_datasets.py --preference --dpo --rlaif

  # Download evaluation datasets (persona, sycophancy, AI risk)
  python download_datasets.py --persona --sycophancy --ai-risk

  # Download for specific enhanced features
  python download_datasets.py --for-moh --for-rag --for-continual-learning

  # Custom data directory and batch size
  python download_datasets.py --all --output-dir /custom/path/data --batch-size 500

  # Download small datasets only
  python download_datasets.py --all --skip-large
        """
    )

    # Output configuration
    parser.add_argument("--output-dir", default="/project/code/data",
                      help="Output directory for datasets (streaming, no caching)")
    parser.add_argument("--max-samples", type=int, default=None,
                      help="Maximum samples per dataset")
    parser.add_argument("--batch-size", type=int, default=10000,
                      help="Batch size for memory-efficient processing (default: 1000)")

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

    # Anthropic-specific categories
    parser.add_argument("--anthropic", action="store_true",
                       help="Download all Anthropic datasets (HH-RLHF, model-written-evals, etc.)")
    parser.add_argument("--constitutional-ai", action="store_true",
                       help="Download Constitutional AI and RLAIF datasets")
    parser.add_argument("--persona", action="store_true",
                       help="Download persona and behavior evaluation datasets")
    parser.add_argument("--sycophancy", action="store_true",
                       help="Download sycophancy detection datasets")
    parser.add_argument("--ai-risk", action="store_true",
                       help="Download AI risk evaluation datasets")
    parser.add_argument("--red-teaming", action="store_true",
                       help="Download red-teaming and adversarial datasets")
    parser.add_argument("--preference", action="store_true",
                       help="Download preference learning datasets (RLHF/DPO)")
    parser.add_argument("--harmlessness", action="store_true",
                       help="Download harmlessness-focused datasets")
    parser.add_argument("--helpfulness", action="store_true",
                       help="Download helpfulness-focused datasets")
    parser.add_argument("--toxicity", action="store_true",
                       help="Download toxicity detection datasets")
    parser.add_argument("--fairness", action="store_true",
                       help="Download fairness and bias detection datasets")
    parser.add_argument("--dpo", action="store_true",
                       help="Download Direct Preference Optimization datasets")
    parser.add_argument("--rlaif", action="store_true",
                       help="Download Reinforcement Learning from AI Feedback datasets")

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

    # Initialize downloader with batch size
    downloader = DatasetDownloader(output_dir=args.output_dir,
                                 max_samples=args.max_samples,
                                 batch_size=args.batch_size)

    # Determine what to download
    datasets = None

    # DEFAULT: Download ALL datasets when no args provided
    if not any([args.dataset, args.all, args.pretraining, args.rag, args.multitask,
                args.continual, args.evaluation, args.safety, args.multimodal,
                args.code, args.conversation, args.anthropic, args.constitutional_ai,
                args.persona, args.sycophancy, args.ai_risk, args.red_teaming,
                args.preference, args.harmlessness, args.helpfulness, args.toxicity,
                args.fairness, args.dpo, args.rlaif, args.for_moh, args.for_moa,
                args.for_rag, args.for_continual_learning, args.for_cross_attention,
                args.for_evaluation, args.for_safety]):
        # No arguments = Download ALL DATASETS by default
        print("\n🎯 DEFAULT: Downloading ALL AVAILABLE DATASETS")
        print("="*70)
        print("This will download all configured datasets for comprehensive LLM training.")
        print("="*70)

        # Get all datasets
        datasets = None  # None means download all
        total_datasets = len(DATASETS_CONFIG)

        # Group by category for display
        by_category = {}
        for name, config in DATASETS_CONFIG.items():
            categories = config.get("categories", ["other"])
            main_cat = categories[0] if categories else "other"
            if main_cat not in by_category:
                by_category[main_cat] = []
            by_category[main_cat].append((name, config))

        print(f"\n📊 Downloading ALL {total_datasets} datasets:")

        # Show top priority datasets
        priority_datasets = []
        for name, config in DATASETS_CONFIG.items():
            priority = config.get("priority", 999)
            if priority <= 3:
                priority_datasets.append((priority, name, config))

        if priority_datasets:
            priority_datasets.sort()
            print("\n🔹 High Priority Datasets:")
            for priority, name, config in priority_datasets[:10]:
                print(f"   • [{priority}] {name}: {config.get('description', '')}")

        print(f"\n📂 Categories included ({len(by_category)} categories):")
        for category, dataset_list in by_category.items():
            print(f"   • {category.upper().replace('_', ' ')}: {len(dataset_list)} datasets")

        print("\n💡 Tips:")
        print("   • Use --anthropic to download only Anthropic datasets")
        print("   • Use --skip-large to skip very large datasets")
        print("   • Use specific flags like --code, --math, etc. for specific categories")
        print("   • Use --max-samples 1000 to limit samples per dataset for testing")
        print("="*70)

    elif args.dataset:
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

        # Anthropic-specific categories
        if args.anthropic:
            categories.extend(["anthropic", "safety", "rlhf", "preference", "harmlessness", "helpfulness"])
        if args.constitutional_ai:
            categories.extend(["constitutional_ai", "rlaif", "harmlessness", "synthetic"])
        if args.persona:
            categories.extend(["persona", "behavior", "evaluation"])
        if args.sycophancy:
            categories.extend(["sycophancy", "evaluation", "bias"])
        if args.ai_risk:
            categories.extend(["ai_risk", "evaluation", "safety"])
        if args.red_teaming:
            categories.extend(["red_teaming", "adversarial", "safety", "toxicity"])
        if args.preference:
            categories.extend(["preference", "rlhf", "dpo", "feedback"])
        if args.harmlessness:
            categories.extend(["harmlessness", "safety", "constitutional_ai"])
        if args.helpfulness:
            categories.extend(["helpfulness", "instruction", "qa"])
        if args.toxicity:
            categories.extend(["toxicity", "safety", "bias", "red_teaming"])
        if args.fairness:
            categories.extend(["fairness", "bias", "safety"])
        if args.dpo:
            categories.extend(["dpo", "preference", "rlhf"])
        if args.rlaif:
            categories.extend(["rlaif", "constitutional_ai", "synthetic"])

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