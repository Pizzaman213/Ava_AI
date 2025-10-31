#!/usr/bin/env python3
"""
Unified Dataset Downloader - Complete merger of all download scripts

This script combines functionality from:
- download_high_quality_simple.py: High-quality dataset selection with JSONL output
- download_stories.py: Story filtering with "once upon a time" detection
- download_datasets.py: Comprehensive 80+ dataset configurations
- download_hello_world_datasets.py: Greeting and conversational datasets
- download_all_greeting_datasets.py: Comprehensive greeting detection and conversational datasets

Features:
- 90+ pre-configured datasets with verified parameters (added 10 new greeting datasets)
- Smart retry strategies with fallback options
- Memory-efficient streaming and batching
- Story filtering (e.g., "once upon a time")
- Greeting/conversational filtering with custom examples
- Enhanced conversation extraction (dialog, utterances, prompt-response formats)
- Quality-based dataset selection
- JSONL output format with greeting metadata
- Default ~10B token high-quality download preset
"""

import os
import sys
import json
import time
import argparse
import traceback
import psutil
import re
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

# Comprehensive dataset configuration
DATASETS_CONFIG = {
    # ================================
    # TIER 1: HIGHEST QUALITY INSTRUCTION DATA
    # ================================
    "teknium/OpenHermes-2.5": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction", "qa"], "tokens": "high",
        "quality_score": 10, "priority": 1,
        "estimated_tokens_millions": 500, "max_samples": None,
        "default_10b": True,
        "description": "1M+ GPT-4 generated Q&A pairs - highest quality instruction dataset"
    },
    "Open-Orca/OpenOrca": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "instruction", "qa"], "tokens": "very_high",
        "quality_score": 9.5, "priority": 1,
        "estimated_tokens_millions": 2000, "max_samples": 2000000,
        "default_10b": True,
        "description": "4M GPT-4/GPT-3.5 instruction Q&A pairs from FLAN"
    },
    "meta-math/MetaMathQA": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "math", "qa"], "tokens": "high",
        "quality_score": 9, "priority": 2,
        "estimated_tokens_millions": 200, "max_samples": None,
        "default_10b": True,
        "description": "395k mathematical Q&A with step-by-step solutions"
    },
    "m-a-p/Code-Feedback": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "code", "qa"], "tokens": "high",
        "quality_score": 8, "priority": 3,
        "estimated_tokens_millions": 300, "max_samples": 300000,
        "default_10b": True,
        "description": "Code Q&A dataset with detailed feedback"
    },
    "OpenAssistant/oasst2": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["pretraining", "conversation", "qa", "greeting"], "tokens": "high",
        "quality_score": 8.5, "priority": 2,
        "estimated_tokens_millions": 100, "max_samples": None,
        "default_10b": True,
        "description": "Enhanced human-ranked conversational Q&A dataset"
    },

    # ================================
    # TIER 2: HIGH-QUALITY WEB TEXT
    # ================================
    "togethercomputer/RedPajama-Data-1T": {
        "splits": ["train"], "subset": "default", "streaming_safe": True,
        "categories": ["rag", "pretraining", "web"], "tokens": "very_high", "large": True,
        "estimated_tokens_millions": 200000, "max_samples": 10000000,
        "default_10b": True, "quality_score": 8.5, "priority": 1,
        "description": "1 trillion token dataset - clean-room LLaMa replication with 7 high-quality sources",
        "subsets": ["arxiv", "c4", "common_crawl", "github", "stackexchange", "wikipedia"]
    },
    "allenai/c4": {
        "splits": ["train"], "subset": "en", "streaming_safe": True,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "estimated_tokens_millions": 5000, "max_samples": 5000000,
        "default_10b": True, "quality_score": 7.5, "priority": 2,
        "description": "Colossal Clean Crawled Corpus - cleaned web text for language modeling"
    },
    "HuggingFaceFW/fineweb": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["rag", "web"], "tokens": "very_high", "large": True,
        "estimated_tokens_millions": 2000, "max_samples": 2000000,
        "default_10b": True, "quality_score": 8, "priority": 2,
        "description": "High-quality web text filtered from CommonCrawl"
    },
    "HuggingFaceFW/fineweb-edu": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["rag", "education"], "tokens": "very_high", "large": True,
        "estimated_tokens_millions": 1000, "max_samples": 1000000,
        "default_10b": True, "quality_score": 8.5, "priority": 1,
        "description": "Educational web content from FineWeb corpus"
    },
    "openwebtext": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["rag", "pretraining"], "tokens": "very_high", "large": True,
        "max_samples": 500000,
        "default_10b": True,
        "description": "Open-source recreation of GPT-2's WebText training dataset"
    },
    "wikipedia": {
        "splits": ["train"], "subset": "20220301.en", "streaming_safe": True,
        "categories": ["rag", "knowledge"], "tokens": "very_high", "large": True,
        "max_samples": 100000,
        "default_10b": True,
        "description": "English Wikipedia articles for knowledge-intensive tasks"
    },

    # ================================
    # TIER 3: CODE DATASETS
    # ================================
    "sahil2801/CodeAlpaca-20k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "instruction"], "tokens": "medium",
        "description": "Code generation and instruction following dataset"
    },
    "iamtarun/python_code_instructions_18k_alpaca": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "python"], "tokens": "medium",
        "default_10b": True,
        "description": "Python-specific coding instructions and solutions"
    },
    "m-a-p/CodeFeedback-Filtered-Instruction": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["code", "feedback"], "tokens": "high",
        "description": "Code instruction dataset with feedback filtering"
    },

    # ================================
    # TIER 4: CONVERSATIONAL & STORY DATASETS
    # ================================
    "HuggingFaceH4/ultrachat_200k": {
        "splits": ["train_sft", "test_sft"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "multiturn", "greeting"], "tokens": "very_high", "large": True,
        "description": "200K high-quality multi-turn conversations",
        "default_10b": True,
        "max_samples": 200000,
    },
    "HuggingFaceH4/no_robots": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "synthetic", "greeting"], "tokens": "high",
        "description": "Human-generated conversations without AI assistance",
        "default_10b": True,
        "max_samples": 10000,
    },
    "Anthropic/hh-rlhf": {
        "splits": ["train", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "anthropic", "safety", "rlhf", "preference"],
        "tokens": "high", "quality_score": 9, "priority": 1,
        "estimated_tokens_millions": 150, "max_samples": 160000,
        "default_10b": True,
        "description": "Anthropic's human preference data for helpful and harmless AI with greetings"
    },
    "OpenAssistant/oasst1": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "qa", "greeting"], "tokens": "high",
        "quality_score": 8,
        "estimated_tokens_millions": 50, "max_samples": 100000,
        "default_10b": True,
        "description": "Human-generated, assistant-ranked conversation Q&A trees"
    },
    "daily_dialog": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "dialog"], "tokens": "medium",
        "max_samples": 13000,
        "default_10b": True,
        "description": "Daily conversations covering greetings and small talk"
    },
    "empathetic_dialogues": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "emotion"], "tokens": "high",
        "max_samples": 25000,
        "default_10b": True,
        "description": "Empathetic conversations with emotional context"
    },
    "AlekseyKorshuk/persona-chat": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "persona"], "tokens": "medium",
        "max_samples": 10000,
        "default_10b": True,
        "description": "Persona-based chit-chat conversations"
    },
    "blended_skill_talk": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "multiskill"], "tokens": "medium",
        "max_samples": 5000,
        "default_10b": True,
        "description": "Multi-skill conversations including greetings"
    },
    "conv_ai_2": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting"], "tokens": "medium",
        "max_samples": 10000,
        "description": "Conversational AI dialogue dataset"
    },
    "google/Synthetic-Persona-Chat": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "synthetic", "persona"], "tokens": "high",
        "max_samples": 50000,
        "description": "Synthetic persona-based conversations"
    },
    "microsoft/wizard_of_wikipedia": {
        "splits": ["train", "test", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "knowledge"], "tokens": "high",
        "max_samples": 20000,
        "description": "Knowledge-grounded conversations from Wizard of Wikipedia"
    },
    "Salesforce/dialogstudio": {
        "splits": ["train"], "subset": "TradeDial", "streaming_safe": True,
        "categories": ["conversation", "greeting", "dialog"], "tokens": "medium",
        "max_samples": 5000,
        "description": "Multi-domain dialogue dataset from Salesforce"
    },
    "AllenAI/prosocial-dialog": {
        "splits": ["train", "validation", "test"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "prosocial"], "tokens": "high",
        "max_samples": 20000,
        "description": "Prosocial dialogue dataset from AllenAI"
    },
    "PygmalionAI/PIPPA": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "roleplay"], "tokens": "high",
        "max_samples": 30000,
        "description": "Personal Interaction Pairs between People and AI"
    },
    "HuggingFaceH4/self-instruct": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "synthetic"], "tokens": "medium",
        "max_samples": 10000,
        "description": "Self-Instruct synthetic instruction dataset"
    },
    "garage-bAInd/Open-Platypus": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "qa"], "tokens": "high",
        "max_samples": 25000,
        "description": "Open-Platypus curated instruction dataset"
    },
    "WizardLM/WizardLM_evol_instruct_V2_196k": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "qa"], "tokens": "very_high",
        "max_samples": 50000,
        "description": "WizardLM evolved instruction dataset V2"
    },
    "fnlp/moss-002-sft-data": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "greeting", "multilingual"], "tokens": "high",
        "max_samples": 20000,
        "description": "MOSS conversational SFT dataset"
    },
    "timdettmers/openassistant-guanaco": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["conversation", "qa", "greeting"], "tokens": "medium",
        "max_samples": 10000,
        "description": "OpenAssistant Guanaco conversational dataset"
    },
    "QingyiSi/Alpaca-CoT": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["instruction", "cot", "qa"], "tokens": "high",
        "max_samples": 30000,
        "description": "Alpaca with Chain-of-Thought annotations"
    },
    "roneneldan/TinyStories": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["synthetic", "stories"], "tokens": "high",
        "default_10b": True,
        "description": "Simple stories for small language models (great for filtering 'once upon a time')"
    },

    # ================================
    # TIER 5: ANTHROPIC & SAFETY DATASETS
    # ================================
    "Anthropic/model-written-evals": {
        "splits": ["train"], "subset": None, "streaming_safe": True,
        "categories": ["anthropic", "evaluation", "persona", "safety"],
        "tokens": "medium", "quality_score": 9, "priority": 1,
        "description": "Anthropic's model-written evaluations for persona, sycophancy, AI risks"
    },
    "HuggingFaceH4/ultrafeedback_binarized": {
        "splits": ["train_prefs", "test_prefs"], "subset": None, "streaming_safe": True,
        "categories": ["safety", "feedback", "preference", "rlhf"],
        "tokens": "high", "quality_score": 8.5, "priority": 2,
        "description": "High-quality preference data for RLHF from GPT-4 feedback"
    },

    # ================================
    # EVALUATION DATASETS
    # ================================
    "allenai/ai2_arc": {
        "splits": ["train", "test", "validation"], "subset": "ARC-Challenge",
        "streaming_safe": True,
        "categories": ["evaluation", "reasoning"], "tokens": "low",
        "description": "AI2 Reasoning Challenge - grade-school science questions"
    },
    "openai/gsm8k": {
        "splits": ["train", "test"], "subset": "main", "streaming_safe": True,
        "categories": ["multitask", "continual", "math"], "tokens": "medium",
        "description": "Grade School Math 8K - math word problems with solutions"
    },
    "rajpurkar/squad": {
        "splits": ["train", "validation"], "subset": None, "streaming_safe": True,
        "categories": ["evaluation", "qa"], "tokens": "medium",
        "description": "Stanford Question Answering Dataset for reading comprehension"
    },
}

# Retry strategies for different failure modes
RETRY_STRATEGIES = [
    {"name": "streaming_mode", "params": {"streaming": True, "cache_dir": None}},
    {"name": "streaming_with_token", "params": {"streaming": True, "token": True, "cache_dir": None}},
    {"name": "streaming_trust_remote", "params": {"streaming": True, "trust_remote_code": True, "cache_dir": None}},
    {"name": "minimal_streaming", "params": {"streaming": True}},
]


class UnifiedDownloader:
    """Unified downloader combining all features"""

    def __init__(self, output_dir: str = "/project/code/data",
                 max_samples: Optional[int] = None,
                 batch_size: int = 1000):
        """Initialize the unified downloader"""
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

    def extract_text(self, sample, dataset_name):
        """Extract text from sample based on dataset structure"""

        # ========== INSTRUCTION DATASETS ==========
        if "ultrachat" in dataset_name.lower():
            if 'messages' in sample and isinstance(sample['messages'], list):
                texts = []
                for msg in sample['messages']:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    texts.append(f"{role.capitalize()}: {content}")
                return '\n'.join(texts)

        if "slimorca" in dataset_name.lower() or "orca" in dataset_name.lower():
            if 'conversations' in sample and isinstance(sample['conversations'], list):
                texts = []
                for msg in sample['conversations']:
                    from_who = msg.get('from', 'unknown')
                    value = msg.get('value', '')
                    role = 'User' if from_who == 'human' else 'Assistant'
                    texts.append(f"{role}: {value}")
                return '\n'.join(texts)
            if 'question' in sample and 'answer' in sample:
                return f"Question: {sample['question']}\nAnswer: {sample['answer']}"
            if 'system' in sample and 'user' in sample and 'assistant' in sample:
                return f"System: {sample['system']}\nUser: {sample['user']}\nAssistant: {sample['assistant']}"

        if "openhermes" in dataset_name.lower():
            if 'conversations' in sample and isinstance(sample['conversations'], list):
                texts = []
                for msg in sample['conversations']:
                    role = msg.get('from', msg.get('role', 'unknown'))
                    value = msg.get('value', msg.get('content', ''))
                    texts.append(f"{role.capitalize()}: {value}")
                return '\n'.join(texts)

        if "sharegpt" in dataset_name.lower():
            if 'conversations' in sample and isinstance(sample['conversations'], list):
                texts = []
                for msg in sample['conversations']:
                    from_who = msg.get('from', 'unknown')
                    value = msg.get('value', '')
                    role = 'User' if from_who in ['human', 'user'] else 'Assistant'
                    texts.append(f"{role}: {value}")
                return '\n'.join(texts)

        if "lima" in dataset_name.lower():
            if 'conversations' in sample and isinstance(sample['conversations'], list):
                texts = []
                for i, turn in enumerate(sample['conversations']):
                    role = 'User' if i % 2 == 0 else 'Assistant'
                    texts.append(f"{role}: {turn}")
                return '\n'.join(texts)

        # ========== WEB TEXT DATASETS ==========
        if "redpajama" in dataset_name.lower():
            if 'text' in sample:
                return sample['text']

        if any(x in dataset_name.lower() for x in ['fineweb', 'c4', 'dolma', 'pile', 'openwebtext']):
            if 'text' in sample:
                return sample['text']

        # ========== CODE DATASETS ==========
        if "codealpaca" in dataset_name.lower() or "code_instructions" in dataset_name.lower():
            if 'instruction' in sample and 'output' in sample:
                instruction = sample.get('instruction', '')
                input_text = sample.get('input', '')
                output = sample.get('output', '')
                if input_text:
                    return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                else:
                    return f"Instruction: {instruction}\nOutput: {output}"

        if "codefeedback" in dataset_name.lower():
            if 'query' in sample and 'answer' in sample:
                return f"Query: {sample['query']}\nAnswer: {sample['answer']}"

        # ========== MATH DATASETS ==========
        if "metamath" in dataset_name.lower() or "gsm8k" in dataset_name.lower():
            # Handle MetaMathQA and similar math datasets
            query = sample.get('query') or sample.get('problem') or sample.get('question')
            response = sample.get('response') or sample.get('solution') or sample.get('answer')

            # Ensure both query and response are valid strings
            if query and response and isinstance(query, str) and isinstance(response, str):
                return f"Problem: {query}\nSolution: {response}"
            elif query and isinstance(query, str):
                return f"Problem: {query}"
            elif response and isinstance(response, str):
                return f"Solution: {response}"

        if "starcoder" in dataset_name.lower() or "stack" in dataset_name.lower():
            if 'content' in sample:
                return sample['content']
            if 'code' in sample:
                return sample['code']

        # ========== KNOWLEDGE DATASETS ==========
        if "wikipedia" in dataset_name.lower() or "book" in dataset_name.lower():
            if 'text' in sample:
                return sample['text']

        # ========== CONVERSATIONAL DATASETS ==========
        if "hh-rlhf" in dataset_name.lower() or "anthropic" in dataset_name.lower():
            if 'chosen' in sample:
                return sample['chosen']

        if "oasst" in dataset_name.lower() or "openassistant" in dataset_name.lower():
            if 'text' in sample:
                return sample['text']

        if "no_robots" in dataset_name.lower():
            if 'messages' in sample and isinstance(sample['messages'], list):
                texts = []
                for msg in sample['messages']:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    texts.append(f"{role.capitalize()}: {content}")
                return '\n'.join(texts)

        # Dialog/dialogue format (daily_dialog, empathetic_dialogues, etc.)
        if 'dialog' in sample or 'dialogue' in sample:
            dialog = sample.get('dialog', sample.get('dialogue', []))
            if isinstance(dialog, list):
                texts = []
                for i, turn in enumerate(dialog):
                    speaker = "User" if i % 2 == 0 else "Assistant"
                    if isinstance(turn, dict):
                        text = turn.get('text', turn.get('utterance', ''))
                        speaker = turn.get('speaker', speaker)
                    else:
                        text = str(turn)
                    if text:
                        texts.append(f"{speaker}: {text}")
                if texts:
                    return '\n'.join(texts)

        # Utterances format (used in some dialogue datasets)
        if 'utterances' in sample:
            utts = sample['utterances']
            if isinstance(utts, list):
                texts = []
                for utt in utts:
                    if isinstance(utt, dict):
                        speaker = utt.get('speaker', utt.get('actor_type', 'Speaker'))
                        text = utt.get('text', utt.get('utterance', ''))
                        if text:
                            texts.append(f"{speaker}: {text}")
                    else:
                        texts.append(str(utt))
                if texts:
                    return '\n'.join(texts)

        # Prompt-response format
        if 'prompt' in sample and 'response' in sample:
            prompt = sample.get('prompt')
            response = sample.get('response')
            if prompt and response and isinstance(prompt, str) and isinstance(response, str):
                return f"User: {prompt}\nAssistant: {response}"

        # Question-answer format
        if 'question' in sample and 'answer' in sample:
            question = sample.get('question')
            answer = sample.get('answer')
            if question and answer and isinstance(question, str) and isinstance(answer, str):
                return f"User: {question}\nAssistant: {answer}"

        # Instruction-response format (with optional context/input)
        if 'instruction' in sample:
            instruction = sample.get('instruction')
            response = sample.get('response', sample.get('output', ''))
            context = sample.get('context', sample.get('input', ''))
            if instruction and isinstance(instruction, str):
                if context and isinstance(context, str):
                    return f"User: {instruction}\nContext: {context}\nAssistant: {response}"
                elif response and isinstance(response, str):
                    return f"User: {instruction}\nAssistant: {response}"

        # ========== GENERIC FALLBACK ==========
        for field in ['text', 'content', 'chosen', 'response', 'output', 'story', 'narrative', 'body']:
            if field in sample and isinstance(sample[field], str) and sample[field]:
                return sample[field]

        # Final fallback: try query + answer combination
        if 'query' in sample and 'answer' in sample:
            query = sample.get('query')
            answer = sample.get('answer')
            if query and answer and isinstance(query, str) and isinstance(answer, str):
                return f"Query: {query}\nAnswer: {answer}"

        # Last resort: combine text fields
        text_parts = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10 and key not in ['id', 'source', 'dataset']:
                text_parts.append(f"{key.capitalize()}: {value}")

        return '\n'.join(text_parts) if text_parts else None

    def starts_with_once_upon(self, text):
        """Check if text starts with variations of 'once upon a time'"""
        if not text:
            return False

        text_lower = text.strip().lower()

        patterns = [
            r'^once upon a time',
            r'^once upon a time,',
            r'^once upon a time there',
            r'^once, upon a time',
            r'^long ago',
            r'^a long time ago',
            r'^many years ago',
            r'^in a land far away',
            r'^there once was',
            r'^there once lived',
        ]

        for pattern in patterns:
            if re.match(pattern, text_lower):
                return True

        return False

    def has_greeting_words(self, text: str) -> bool:
        """Check if text contains greeting words - matches download_all_greeting_datasets.py logic."""
        if not text:
            return False

        text_lower = text.lower()

        # Common greetings - comprehensive list from download_all_greeting_datasets.py
        greetings = [
            'hello', 'hi ', 'hi,', 'hi!', 'hi.', 'hey', 'greetings',
            'good morning', 'good afternoon', 'good evening', 'howdy',
            "what's up", 'whats up', 'how are you', 'how do you do',
            'nice to meet', 'pleased to meet', 'welcome', 'hiya',
            'hola', 'bonjour', 'ciao', 'aloha', 'namaste',
            'salutations', 'sup ', 'yo ', 'heya'
        ]

        return any(greeting in text_lower for greeting in greetings)

    def create_custom_greeting_conversations(self):
        """Create custom conversational examples with greetings."""
        processed_path = self.output_dir / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        output_file = processed_path / "custom_greeting_conversations.jsonl"

        conversations = [
            # Basic greetings
            {
                "text": "User: Hello! How are you today?\nAssistant: Hello! I'm doing well, thank you for asking. How can I help you today?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Hi there!\nAssistant: Hi! Welcome! What brings you here today?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Good morning!\nAssistant: Good morning! I hope you're having a great start to your day. What can I assist you with?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Hey, what's up?\nAssistant: Hey! Not much, just here to help. What's on your mind?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Greetings!\nAssistant: Greetings to you as well! How may I be of service today?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Multi-turn conversations with greetings
            {
                "text": "User: Hello!\nAssistant: Hello! Nice to meet you!\nUser: Nice to meet you too! Can you help me with something?\nAssistant: Of course! I'd be happy to help. What do you need?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Hi, I'm new here.\nAssistant: Hi! Welcome! It's great to have you. Is there anything you'd like to know or any way I can help you get started?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Good evening!\nAssistant: Good evening! I hope you've had a pleasant day. What can I do for you this evening?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Contextual greetings
            {
                "text": "User: Hello, I'm looking for some help with Python.\nAssistant: Hello! I'd be glad to help you with Python. What specific topic or problem are you working on?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Hi! Can you explain how to use variables?\nAssistant: Hi there! Absolutely, I can explain variables. In programming, a variable is like a container that stores data. Would you like me to show you some examples?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Casual greetings
            {
                "text": "User: Yo!\nAssistant: Hey! What can I help you with today?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Howdy!\nAssistant: Howdy! Great to see you! What brings you by?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Formal greetings
            {
                "text": "User: Good day! I hope this message finds you well.\nAssistant: Good day to you! Thank you for the kind words. I'm here and ready to assist. How may I help you?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Pleased to make your acquaintance.\nAssistant: The pleasure is mine! I'm delighted to meet you. What can I assist you with today?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # International greetings
            {
                "text": "User: Hola! Do you speak English?\nAssistant: Hello! Yes, I speak English. How can I help you today?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Bonjour!\nAssistant: Hello! While I primarily speak English, I'm happy to help you. What do you need?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Time-specific greetings
            {
                "text": "User: Good morning! I have a question about data structures.\nAssistant: Good morning! I'd love to help you understand data structures. Which one are you curious about?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Good afternoon! Quick question.\nAssistant: Good afternoon! Sure, I'm all ears. What's your question?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Follow-up greetings
            {
                "text": "User: Hi again!\nAssistant: Hello again! Welcome back! What can I help you with this time?",
                "source": "custom_greetings",
                "has_greeting": True
            },
            {
                "text": "User: Hey, it's me again.\nAssistant: Hey! Good to see you back! What would you like to work on today?",
                "source": "custom_greetings",
                "has_greeting": True
            },

            # Extended conversations
            {
                "text": "User: Hello!\nAssistant: Hello! How are you doing today?\nUser: I'm doing great, thanks! How about you?\nAssistant: I'm doing well, thank you for asking! I'm here and ready to help. What would you like to talk about or work on?",
                "source": "custom_greetings",
                "has_greeting": True
            },
        ]

        # Add variations
        greetings_list = ["Hello", "Hi", "Hey", "Greetings", "Good day", "Howdy"]
        responses = [
            "I'm doing well, thank you!",
            "I'm great, thanks for asking!",
            "Pretty good! How about you?",
            "I'm fine, thanks!",
            "Doing wonderful, thank you!"
        ]

        for i, greeting in enumerate(greetings_list):
            for j, response in enumerate(responses[:3]):  # Use first 3 responses
                conversations.append({
                    "text": f"User: {greeting}! How are you?\nAssistant: {greeting}! {response} How can I help you today?",
                    "source": "custom_greetings",
                    "has_greeting": True
                })

        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ Created custom greeting conversations: {output_file.name}")
        print(f"  Total conversations: {len(conversations)}")
        print(f"  File size: {size_mb:.2f} MB")

        return len(conversations)

    def download_dataset_with_retry(self, dataset_name: str, config: Dict,
                                   filter_stories: bool = False,
                                   filter_greetings: bool = False) -> bool:
        """Download a dataset with multiple retry strategies"""
        print(f"\n{'='*60}")
        print(f"📥 Downloading: {dataset_name}")
        if filter_stories:
            print(f"   Filtering for 'once upon a time' stories")
        if filter_greetings:
            print(f"   Filtering for greeting conversations")
        print(f"{'='*60}")

        is_large = config.get("large", False)

        # Check memory for large datasets
        available_gb, _ = self.check_memory()
        if is_large and available_gb < 10:
            print(f"⚠️  Low memory ({available_gb:.1f}GB available), using streaming mode")

        # Track if any file was successfully saved
        any_saved = False

        # Try different strategies
        for strategy_idx, strategy in enumerate(RETRY_STRATEGIES):
            print(f"\nAttempt {strategy_idx + 1}/{len(RETRY_STRATEGIES)}: {strategy['name']}")

            try:
                params = strategy["params"].copy()

                if config.get("subset"):
                    dataset_args = [dataset_name, config["subset"]]
                else:
                    dataset_args = [dataset_name]

                # Always use streaming
                params["streaming"] = True
                params["cache_dir"] = None

                # Handle different splits
                for split in config.get("splits", ["train"]):
                    print(f"  Processing split: {split}")

                    try:
                        if strategy_idx > 0:
                            time.sleep(2)

                        dataset = None
                        try:
                            dataset = self.load_dataset(*dataset_args, split=split, **params)
                        except Exception as load_error:
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

                        # Create output directories
                        processed_path = self.output_dir / "processed"
                        processed_path.mkdir(parents=True, exist_ok=True)

                        safe_name = dataset_name.replace("/", "_")
                        if filter_stories:
                            jsonl_file = processed_path / f"{safe_name}_once_upon_stories.jsonl"
                        elif filter_greetings:
                            jsonl_file = processed_path / f"{safe_name}_conversational.jsonl"
                        else:
                            jsonl_file = processed_path / f"{safe_name}_processed.jsonl"

                        # Skip if already exists
                        if jsonl_file.exists():
                            existing_lines = sum(1 for _ in open(jsonl_file))
                            print(f"  ⚠️  Already exists with {existing_lines:,} examples - SKIPPING")
                            if dataset_name not in self.summary["skipped"]:
                                self.summary["skipped"].append(dataset_name)
                            return True

                        # Stream and save samples
                        saved = 0
                        processed_count = 0
                        max_to_download = self.max_samples if self.max_samples else config.get("max_samples", 100000)
                        if max_to_download is None:
                            max_to_download = 100000

                        print(f"  Streaming up to {max_to_download:,} samples...")

                        with open(jsonl_file, 'w', encoding='utf-8') as jf:
                            for idx, sample in enumerate(tqdm(dataset, total=max_to_download, desc=f"{dataset_name}")):
                                processed_count += 1

                                # Extract text
                                text = self.extract_text(sample, dataset_name)

                                if text and len(text) > 20:
                                    # Apply story filter if requested
                                    if filter_stories:
                                        if not self.starts_with_once_upon(text):
                                            continue

                                    # Apply greeting filter if requested
                                    has_greeting = False
                                    if filter_greetings:
                                        has_greeting = self.has_greeting_words(text)
                                        # Keep only 30% non-greeting in greeting mode
                                        if not has_greeting and saved >= max_to_download * 0.3:
                                            continue

                                    # Save to JSONL
                                    record = {
                                        'text': text,
                                        'source': dataset_name,
                                        'type': 'story' if filter_stories else 'conversation' if filter_greetings else 'general'
                                    }
                                    if filter_greetings:
                                        record['has_greeting'] = has_greeting if filter_greetings else self.has_greeting_words(text)

                                    jf.write(json.dumps(record, ensure_ascii=False) + '\n')
                                    saved += 1

                                if saved >= max_to_download:
                                    break

                                # Safety limit for filtered datasets
                                if filter_stories and processed_count > max_to_download * 10:
                                    print(f"  ⚠️  Processed {processed_count:,}, found {saved:,} matching stories")
                                    break

                                # Memory check every 10k samples
                                if processed_count % 10000 == 0:
                                    available_gb, usage_percent = self.check_memory()
                                    if usage_percent > 85:
                                        print(f"  ⚠️  High memory usage ({usage_percent:.1f}%), pausing...")
                                        time.sleep(1)

                        if saved > 0:
                            print(f"  ✅ Saved {saved:,} examples to {jsonl_file.name}")
                            any_saved = True
                            if filter_stories:
                                print(f"  📊 Processed {processed_count:,} total samples")
                        else:
                            # If no samples were saved, remove the empty file
                            if jsonl_file.exists():
                                jsonl_file.unlink()
                            print(f"  ⚠️  No samples saved for split {split}")

                    except Exception as e:
                        print(f"  ✗ Failed to download split {split}: {str(e)}")
                        continue

                # Success if any file was saved
                if any_saved:
                    self.summary["successful"].append(dataset_name)
                    return True

            except Exception as e:
                print(f"  ✗ Strategy failed: {str(e)}")
                continue

        # All strategies failed
        print(f"\n✗ Failed to download {dataset_name} after all attempts")
        self.summary["failed"].append(dataset_name)

        error_file = self.output_dir / ".errors" / f"{dataset_name.replace('/', '_')}.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"All strategies failed\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")

        return False

    def download_all(self, datasets: Optional[List[str]] = None,
                    parallel: bool = False, filter_stories: bool = False,
                    filter_greetings: bool = False, create_custom_greetings: bool = False,
                    max_workers: int = 2):
        """Download all configured datasets"""

        if datasets:
            dataset_configs = {k: v for k, v in DATASETS_CONFIG.items() if k in datasets}
        else:
            dataset_configs = DATASETS_CONFIG

        print(f"\n{'='*60}")
        print(f"UNIFIED DATASET DOWNLOADER")
        print(f"{'='*60}")
        print(f"Datasets to process: {len(dataset_configs)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Parallel mode: {parallel}")
        if filter_stories:
            print(f"Story filtering: ENABLED (once upon a time)")
        if filter_greetings:
            print(f"Greeting filtering: ENABLED")
        if create_custom_greetings:
            print(f"Custom greetings: ENABLED")
        print(f"{'='*60}\n")

        # Create custom greeting conversations if requested
        if create_custom_greetings:
            self.create_custom_greeting_conversations()

        if parallel and len(dataset_configs) > 1:
            # Parallel download
            print(f"Using {max_workers} parallel workers\n")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.download_dataset_with_retry, name, config, filter_stories, filter_greetings): name
                    for name, config in dataset_configs.items()
                }

                for future in as_completed(futures):
                    dataset_name = futures[future]
                    try:
                        success = future.result(timeout=3600)
                        if success:
                            print(f"✓ Completed: {dataset_name}")
                        else:
                            print(f"✗ Failed: {dataset_name}")
                    except Exception as e:
                        print(f"✗ Exception downloading {dataset_name}: {e}")
                        self.summary["failed"].append(dataset_name)
        else:
            # Sequential download
            for dataset_name, config in dataset_configs.items():
                self.download_dataset_with_retry(dataset_name, config, filter_stories, filter_greetings)

        self.save_summary()

    def save_summary(self):
        """Save download summary"""
        self.summary["total_attempted"] = len(self.summary["successful"]) + len(self.summary["failed"])
        self.summary["total_datasets"] = len(DATASETS_CONFIG)

        if self.summary["total_attempted"] > 0:
            self.summary["success_rate"] = len(self.summary["successful"]) / self.summary["total_attempted"]
        else:
            self.summary["success_rate"] = 0

        available_gb, usage_percent = self.check_memory()
        self.summary["memory_stats"] = {
            "available_gb": available_gb,
            "usage_percent": usage_percent
        }

        summary_file = self.output_dir / "download_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.summary, f, indent=2)

        # Count files and estimate tokens
        processed_dir = self.output_dir / "processed"
        total_examples = 0
        total_size = 0

        if processed_dir.exists():
            for jsonl_file in processed_dir.glob("*.jsonl"):
                try:
                    lines = sum(1 for _ in open(jsonl_file))
                    total_examples += lines
                    total_size += jsonl_file.stat().st_size
                except:
                    pass

        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"Total datasets available: {len(DATASETS_CONFIG)}")
        print(f"Successfully downloaded: {len(self.summary['successful'])}")
        print(f"Failed: {len(self.summary['failed'])}")
        print(f"Skipped (existing): {len(self.summary['skipped'])}")
        print(f"Success rate: {self.summary['success_rate']:.1%}")
        print(f"\nTotal examples: {total_examples:,}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")

        # Estimate tokens
        avg_tokens_per_example = 250
        total_tokens = total_examples * avg_tokens_per_example
        print(f"Estimated tokens: ~{total_tokens/1e6:.0f}M ({total_tokens/1e9:.2f}B)")

        print(f"\nSummary saved to: {summary_file}")

        if self.summary["successful"]:
            print(f"\n✅ Successfully downloaded:")
            for name in self.summary["successful"]:
                print(f"  - {name}")

        if self.summary["failed"]:
            print(f"\n❌ Failed downloads:")
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


def main():
    parser = argparse.ArgumentParser(
        description="Unified Dataset Downloader - Merges all download scripts into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DEFAULT: Download ALL available datasets
  python unified_download.py

  # Download specific dataset only
  python unified_download.py --dataset "teknium/OpenHermes-2.5"

  # Download with story filtering (once upon a time)
  python unified_download.py --dataset "roneneldan/TinyStories" --filter-stories

  # Download greeting/conversational datasets
  python unified_download.py --greeting --create-custom-greetings
  python unified_download.py --conversation --filter-greetings

  # Download by category
  python unified_download.py --code --math
  python unified_download.py --anthropic --safety

  # Download with parallel processing
  python unified_download.py --parallel --max-workers 4

  # Custom limits
  python unified_download.py --max-samples 10000 --batch-size 5000
        """
    )

    # Output configuration
    parser.add_argument("--output-dir", default="/project/code/data",
                       help="Output directory for datasets")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for processing")

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
    parser.add_argument("--code", action="store_true",
                       help="Download code datasets")
    parser.add_argument("--math", action="store_true",
                       help="Download math datasets")
    parser.add_argument("--conversation", action="store_true",
                       help="Download conversational datasets")
    parser.add_argument("--greeting", action="store_true",
                       help="Download greeting/hello world datasets")
    parser.add_argument("--anthropic", action="store_true",
                       help="Download Anthropic datasets")
    parser.add_argument("--safety", action="store_true",
                       help="Download safety datasets")
    parser.add_argument("--stories", action="store_true",
                       help="Download story datasets")

    # Special features
    parser.add_argument("--parallel", action="store_true",
                       help="Download datasets in parallel")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Number of parallel workers")
    parser.add_argument("--filter-stories", action="store_true",
                       help="Filter for 'once upon a time' style stories")
    parser.add_argument("--filter-greetings", action="store_true",
                       help="Filter for greeting conversations")
    parser.add_argument("--create-custom-greetings", action="store_true",
                       help="Create custom greeting conversation examples")

    args = parser.parse_args()

    # Initialize downloader
    downloader = UnifiedDownloader(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )

    # Determine what to download
    datasets = None

    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        datasets = None
    else:
        # Category-based or default
        categories = []

        if args.pretraining:
            categories.extend(["pretraining", "instruction"])
        if args.rag:
            categories.extend(["rag", "knowledge", "web"])
        if args.code:
            categories.extend(["code", "python"])
        if args.math:
            categories.extend(["math"])
        if args.conversation:
            categories.extend(["conversation", "dialog"])
        if args.greeting:
            categories.extend(["greeting"])
        if args.anthropic:
            categories.extend(["anthropic"])
        if args.safety:
            categories.extend(["safety", "rlhf"])
        if args.stories:
            categories.extend(["stories", "synthetic"])

        if categories:
            filtered_config = filter_datasets_by_categories(categories)
            datasets = list(filtered_config.keys())
            print(f"\n📊 Found {len(datasets)} datasets matching categories: {set(categories)}")
        else:
            # Default: Download ALL datasets
            datasets = None  # None means all datasets in download_all()
            print(f"\n🎯 DEFAULT MODE: Downloading ALL {len(DATASETS_CONFIG)} Available Datasets")
            total_tokens_m = sum(config.get("estimated_tokens_millions", 0)
                                for config in DATASETS_CONFIG.values())
            print(f"   Estimated: ~{total_tokens_m/1000:.1f}B tokens\n")

    # Start download
    downloader.download_all(
        datasets=datasets,
        parallel=args.parallel,
        filter_stories=args.filter_stories,
        filter_greetings=args.filter_greetings,
        create_custom_greetings=args.create_custom_greetings,
        max_workers=args.max_workers
    )

    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
