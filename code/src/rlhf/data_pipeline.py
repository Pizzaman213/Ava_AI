"""
Comprehensive Data Pipeline for RLHF Training
Handles all dataset types for reasoning, preferences, and self-supervised learning
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
import logging
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset as HFDataset
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict
import pickle
from multiprocessing import Pool
import asyncio
import aiofiles
from LLM.src.utils.path_utils import get_data_dir, get_cache_dir

logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline"""
    # Dataset paths and sources
    preference_datasets: List[str] = field(default_factory=lambda: [
        "Anthropic/hh-rlhf",
        "openai/summarize_from_feedback",
        "Dahoas/synthetic-instruct-gptj-pairwise"
    ])
    
    reasoning_datasets: List[str] = field(default_factory=lambda: [
        "gsm8k", "math_dataset", "competition_math",
        "bigbench", "mmlu", "arc", "hellaswag"
    ])
    
    code_reasoning_datasets: List[str] = field(default_factory=lambda: [
        "deepmind/code_contests", "openai_humaneval", "mbpp"
    ])
    
    scientific_datasets: List[str] = field(default_factory=lambda: [
        "sciq", "ai2_arc", "pubmedqa"
    ])
    
    # Processing settings
    max_length: int = 2048
    chunk_size: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Data augmentation
    augment_preferences: bool = True
    augment_reasoning: bool = True
    paraphrase_prob: float = 0.3
    
    # Caching
    cache_dir: str = None
    use_cache: bool = True
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(get_cache_dir(), "data_pipeline")
    
    # Sampling
    preference_sample_ratio: float = 0.4
    reasoning_sample_ratio: float = 0.3
    self_generated_ratio: float = 0.3

class PreferenceDataset(Dataset):
    """Dataset for human preference data"""
    def __init__(
        self,
        data_sources: List[str],
        tokenizer: Any,
        config: DataPipelineConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        
        # Load data from multiple sources
        for source in data_sources:
            logger.info(f"Loading preference data from {source}")
            self._load_source(source)
        
        logger.info(f"Loaded {len(self.data)} preference examples")
    
    def _load_source(self, source: str):
        """Load data from a specific source"""
        try:
            if source == "Anthropic/hh-rlhf":
                dataset = load_dataset(source, split="train")
                for item in dataset:
                    self.data.append({
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                        "source": source
                    })
            
            elif source == "openai/summarize_from_feedback":
                dataset = load_dataset(source, split="train")
                for item in dataset:
                    self.data.append({
                        "prompt": item["prompt"]["text"],
                        "chosen": item["chosen"]["text"],
                        "rejected": item["rejected"]["text"],
                        "source": source
                    })
            
            elif Path(source).exists():
                # Local file
                with open(source, 'r') as f:
                    if source.endswith('.jsonl'):
                        for line in f:
                            item = json.loads(line)
                            self.data.append(item)
                    else:
                        data = json.load(f)
                        self.data.extend(data)
        
        except Exception as e:
            logger.warning(f"Failed to load {source}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Tokenize
        prompt_enc = self.tokenizer(
            item["prompt"],
            max_length=self.config.max_length // 4,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        chosen_enc = self.tokenizer(
            item["prompt"] + " " + item["chosen"],
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_enc = self.tokenizer(
            item["prompt"] + " " + item["rejected"],
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt_ids": prompt_enc["input_ids"].squeeze(),
            "prompt_mask": prompt_enc["attention_mask"].squeeze(),
            "chosen_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_mask": rejected_enc["attention_mask"].squeeze(),
            "prompt_text": item["prompt"],
            "chosen_text": item["chosen"],
            "rejected_text": item["rejected"],
            "source": item["source"]
        }

class ReasoningDataset(Dataset):
    """Dataset for multi-step reasoning tasks"""
    def __init__(
        self,
        data_sources: List[str],
        tokenizer: Any,
        config: DataPipelineConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []
        
        # Load reasoning datasets
        for source in data_sources:
            logger.info(f"Loading reasoning data from {source}")
            self._load_reasoning_source(source)
        
        logger.info(f"Loaded {len(self.data)} reasoning examples")
    
    def _load_reasoning_source(self, source: str):
        """Load reasoning dataset"""
        try:
            if source == "gsm8k":
                dataset = load_dataset("gsm8k", "main", split="train")
                for item in dataset:
                    self.data.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "reasoning_steps": self._extract_gsm8k_steps(item["answer"]),
                        "type": "math",
                        "source": source
                    })
            
            elif source == "math_dataset":
                # Load mathematical reasoning
                dataset = load_dataset("math_dataset", split="train")
                for item in dataset:
                    self.data.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "reasoning_steps": self._extract_math_steps(item["solution"]),
                        "type": "math",
                        "source": source
                    })
            
            elif "code" in source.lower():
                # Code reasoning datasets
                dataset = load_dataset(source, split="train")
                for item in dataset:
                    self.data.append({
                        "question": item.get("problem", item.get("question", "")),
                        "answer": item.get("solution", item.get("code", "")),
                        "reasoning_steps": self._extract_code_steps(item),
                        "type": "code",
                        "source": source
                    })
            
            elif Path(source).exists():
                # Local file with reasoning data
                with open(source, 'r') as f:
                    data = json.load(f) if source.endswith('.json') else [json.loads(line) for line in f]
                    self.data.extend(data)
        
        except Exception as e:
            logger.warning(f"Failed to load reasoning source {source}: {e}")
    
    def _extract_gsm8k_steps(self, answer: str) -> List[str]:
        """Extract reasoning steps from GSM8K answer"""
        # GSM8K answers contain step-by-step solutions
        steps = []
        
        # Split by newlines and filter
        lines = answer.strip().split('\n')
        current_step = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new step
            if re.match(r'^(Step \d+|First|Second|Third|Next|Then|Finally)', line, re.IGNORECASE):
                if current_step:
                    steps.append(' '.join(current_step))
                current_step = [line]
            else:
                current_step.append(line)
        
        if current_step:
            steps.append(' '.join(current_step))
        
        # If no clear steps, split by sentences
        if not steps:
            steps = [s.strip() for s in answer.split('.') if s.strip()]
        
        return steps
    
    def _extract_math_steps(self, solution: str) -> List[str]:
        """Extract steps from mathematical solution"""
        steps = []
        
        # Look for equation patterns
        equations = re.findall(r'.*?(?:=|\\implies|\\Rightarrow).*?(?=\n|$)', solution)
        if equations:
            steps.extend(equations)
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', solution, re.DOTALL)
        if numbered_steps:
            steps.extend([s.strip() for s in numbered_steps])
        
        # Fallback to sentence splitting
        if not steps:
            steps = [s.strip() for s in re.split(r'[.!?]+', solution) if s.strip()]
        
        return steps
    
    def _extract_code_steps(self, item: Dict[str, Any]) -> List[str]:
        """Extract reasoning steps from code solution"""
        steps = []
        
        # Extract comments as steps
        code = item.get("solution", item.get("code", ""))
        comments = re.findall(r'(?:#|//|/\*)\s*(.+?)(?:\n|\*/|$)', code)
        steps.extend(comments)
        
        # Extract function definitions
        functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', code)
        for func in functions:
            steps.append(f"Define function {func}")
        
        # Extract algorithm steps from description if available
        if "description" in item:
            algo_steps = self._extract_algorithm_steps(item["description"])
            steps.extend(algo_steps)
        
        return steps
    
    def _extract_algorithm_steps(self, description: str) -> List[str]:
        """Extract algorithmic steps from description"""
        steps = []
        
        # Look for numbered lists
        numbered = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', description, re.DOTALL)
        if numbered:
            steps.extend([s.strip() for s in numbered])
        
        # Look for bullet points
        bullets = re.findall(r'[-*•]\s*(.+?)(?=[-*•]|$)', description, re.DOTALL)
        if bullets:
            steps.extend([s.strip() for s in bullets])
        
        return steps
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Format as reasoning chain
        prompt = f"Question: {item['question']}\n\nLet's solve this step by step:"
        
        # Create reasoning chain text
        reasoning_text = ""
        for i, step in enumerate(item["reasoning_steps"]):
            reasoning_text += f"\nStep {i+1}: {step}"
        reasoning_text += f"\n\nFinal Answer: {item['answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt + reasoning_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Separate prompt and response for training
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.config.max_length // 3,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "prompt_ids": prompt_enc["input_ids"].squeeze(),
            "prompt_length": len(prompt_enc["input_ids"].squeeze()),
            "question": item["question"],
            "answer": item["answer"],
            "reasoning_steps": item["reasoning_steps"],
            "num_steps": len(item["reasoning_steps"]),
            "type": item["type"],
            "source": item["source"]
        }

class SelfGeneratedDataset(IterableDataset):
    """Dataset for self-generated training data"""
    def __init__(
        self,
        generator_model: Any,
        tokenizer: Any,
        config: DataPipelineConfig,
        prompts: List[str]
    ):
        self.generator_model = generator_model
        self.tokenizer = tokenizer
        self.config = config
        self.prompts = prompts
        self.generated_cache = []
    
    def generate_batch(self, batch_size: int = 8) -> List[Dict[str, Any]]:
        """Generate a batch of self-supervised examples"""
        batch_prompts = random.sample(self.prompts, min(batch_size, len(self.prompts)))
        generated_data = []
        
        for prompt in batch_prompts:
            # Generate multiple responses
            responses = []
            quality_scores = []
            
            for _ in range(3):  # Generate 3 variations
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.generator_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.8,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Simple quality score based on perplexity
                    scores = torch.stack(outputs.scores, dim=1)
                    probs = torch.softmax(scores, dim=-1)
                    token_probs = probs.gather(-1, outputs.sequences[:, 1:].unsqueeze(-1))
                    perplexity = torch.exp(-token_probs.log().mean())
                    
                    response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    responses.append(response)
                    quality_scores.append(1.0 / (1.0 + perplexity.item()))
            
            # Create preference pairs
            best_idx = np.argmax(quality_scores)
            worst_idx = np.argmin(quality_scores)
            
            if best_idx != worst_idx:
                generated_data.append({
                    "prompt": prompt,
                    "chosen": responses[best_idx],
                    "rejected": responses[worst_idx],
                    "quality_gap": quality_scores[best_idx] - quality_scores[worst_idx],
                    "source": "self_generated"
                })
        
        return generated_data
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through self-generated data"""
        while True:
            # Generate new batch if cache is empty
            if not self.generated_cache:
                self.generated_cache = self.generate_batch()
            
            # Yield from cache
            if self.generated_cache:
                item = self.generated_cache.pop(0)
                
                # Tokenize
                prompt_enc = self.tokenizer(
                    item["prompt"],
                    max_length=self.config.max_length // 4,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                chosen_enc = self.tokenizer(
                    item["prompt"] + " " + item["chosen"],
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                rejected_enc = self.tokenizer(
                    item["prompt"] + " " + item["rejected"],
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                yield {
                    "prompt_ids": prompt_enc["input_ids"].squeeze(),
                    "prompt_mask": prompt_enc["attention_mask"].squeeze(),
                    "chosen_ids": chosen_enc["input_ids"].squeeze(),
                    "chosen_mask": chosen_enc["attention_mask"].squeeze(),
                    "rejected_ids": rejected_enc["input_ids"].squeeze(),
                    "rejected_mask": rejected_enc["attention_mask"].squeeze(),
                    "quality_gap": item["quality_gap"],
                    "source": item["source"]
                }

class DataAugmentor:
    """Augments training data with various techniques"""
    def __init__(self, config: DataPipelineConfig):
        self.config = config
    
    def augment_preference_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> List[Dict[str, str]]:
        """Augment a preference pair"""
        augmented = []
        
        # Original pair
        augmented.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
        if random.random() < self.config.paraphrase_prob:
            # Paraphrase prompt
            paraphrased_prompt = self._paraphrase_text(prompt)
            augmented.append({
                "prompt": paraphrased_prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        # Swap with confidence adjustment
        if self._is_close_quality(chosen, rejected):
            augmented.append({
                "prompt": prompt,
                "chosen": rejected,
                "rejected": chosen,
                "confidence": 0.6  # Lower confidence for swapped pairs
            })
        
        return augmented
    
    def augment_reasoning_chain(
        self,
        question: str,
        steps: List[str],
        answer: str
    ) -> List[Dict[str, Any]]:
        """Augment reasoning chain data"""
        augmented = []
        
        # Original
        augmented.append({
            "question": question,
            "steps": steps,
            "answer": answer
        })
        
        # Reorder middle steps (if safe)
        if len(steps) > 3:
            reordered_steps = steps.copy()
            middle = reordered_steps[1:-1]
            random.shuffle(middle)
            reordered_steps[1:-1] = middle
            
            augmented.append({
                "question": question,
                "steps": reordered_steps,
                "answer": answer,
                "augmentation": "reordered"
            })
        
        # Add intermediate explanations
        detailed_steps = []
        for i, step in enumerate(steps):
            detailed_steps.append(step)
            if i < len(steps) - 1:
                explanation = self._generate_step_explanation(step, steps[i+1])
                if explanation:
                    detailed_steps.append(f"Explanation: {explanation}")
        
        augmented.append({
            "question": question,
            "steps": detailed_steps,
            "answer": answer,
            "augmentation": "detailed"
        })
        
        return augmented
    
    def _paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing through word substitution"""
        words = text.split()
        
        # Substitute some words with synonyms
        synonym_map = {
            "find": "determine",
            "calculate": "compute",
            "show": "demonstrate",
            "prove": "verify",
            "explain": "describe",
            "solve": "resolve"
        }
        
        paraphrased = []
        for word in words:
            lower_word = word.lower()
            if lower_word in synonym_map and random.random() < 0.5:
                paraphrased.append(word.replace(lower_word, synonym_map[lower_word]))
            else:
                paraphrased.append(word)
        
        return " ".join(paraphrased)
    
    def _is_close_quality(self, text1: str, text2: str) -> bool:
        """Check if two texts are close in quality"""
        # Simple heuristic based on length and complexity
        len_ratio = len(text1) / (len(text2) + 1)
        return 0.8 < len_ratio < 1.2
    
    def _generate_step_explanation(self, current_step: str, next_step: str) -> Optional[str]:
        """Generate explanation between steps"""
        # Simple template-based explanation
        templates = [
            "This leads us to",
            "From this, we can deduce",
            "Therefore",
            "Which gives us",
            "This allows us to"
        ]
        
        if random.random() < 0.3:
            return f"{random.choice(templates)} the next step"
        return None

class UnifiedDataPipeline:
    """Unified data pipeline combining all data sources"""
    def __init__(
        self,
        tokenizer: Any,
        config: DataPipelineConfig,
        generator_model: Optional[Any] = None
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.generator_model = generator_model
        
        # Initialize datasets
        self.preference_dataset = PreferenceDataset(
            config.preference_datasets,
            tokenizer,
            config
        )
        
        self.reasoning_dataset = ReasoningDataset(
            config.reasoning_datasets + config.code_reasoning_datasets + config.scientific_datasets,
            tokenizer,
            config
        )
        
        # Self-generated dataset (if generator provided)
        if generator_model:
            # Extract prompts from existing datasets
            prompts = self._extract_prompts()
            self.self_generated_dataset = SelfGeneratedDataset(
                generator_model,
                tokenizer,
                config,
                prompts
            )
        else:
            self.self_generated_dataset = None
        
        # Data augmentor
        self.augmentor = DataAugmentor(config)
        
        # Cache directory
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _extract_prompts(self) -> List[str]:
        """Extract prompts from existing datasets"""
        prompts = []
        
        # From preference data
        for i in range(min(1000, len(self.preference_dataset))):
            item = self.preference_dataset.data[i]
            prompts.append(item["prompt"])
        
        # From reasoning data
        for i in range(min(1000, len(self.reasoning_dataset))):
            item = self.reasoning_dataset.data[i]
            prompts.append(item["question"])
        
        return prompts
    
    def create_mixed_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create dataloader with mixed data types"""
        # Create a mixed dataset
        mixed_data = []
        
        # Sample from each dataset according to ratios
        num_preference = int(len(self.preference_dataset) * self.config.preference_sample_ratio)
        num_reasoning = int(len(self.reasoning_dataset) * self.config.reasoning_sample_ratio)
        
        # Add preference data
        preference_indices = random.sample(
            range(len(self.preference_dataset)),
            min(num_preference, len(self.preference_dataset))
        )
        
        for idx in preference_indices:
            item = self.preference_dataset[idx]
            item["data_type"] = "preference"
            mixed_data.append(item)
        
        # Add reasoning data
        reasoning_indices = random.sample(
            range(len(self.reasoning_dataset)),
            min(num_reasoning, len(self.reasoning_dataset))
        )
        
        for idx in reasoning_indices:
            item = self.reasoning_dataset[idx]
            item["data_type"] = "reasoning"
            mixed_data.append(item)
        
        # Create custom dataset
        class MixedDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        mixed_dataset = MixedDataset(mixed_data)
        
        return DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self._mixed_collate_fn
        )
    
    def _mixed_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for mixed data types"""
        # Separate by data type
        preference_batch = []
        reasoning_batch = []
        
        for item in batch:
            if item["data_type"] == "preference":
                preference_batch.append(item)
            else:
                reasoning_batch.append(item)
        
        collated = {
            "preference_batch": self._collate_preference(preference_batch) if preference_batch else None,
            "reasoning_batch": self._collate_reasoning(reasoning_batch) if reasoning_batch else None,
        }
        
        return collated
    
    def _collate_preference(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate preference data"""
        return {
            "prompt_ids": torch.stack([item["prompt_ids"] for item in batch]),
            "prompt_mask": torch.stack([item["prompt_mask"] for item in batch]),
            "chosen_ids": torch.stack([item["chosen_ids"] for item in batch]),
            "chosen_mask": torch.stack([item["chosen_mask"] for item in batch]),
            "rejected_ids": torch.stack([item["rejected_ids"] for item in batch]),
            "rejected_mask": torch.stack([item["rejected_mask"] for item in batch]),
        }
    
    def _collate_reasoning(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate reasoning data"""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "prompt_length": torch.tensor([item["prompt_length"] for item in batch]),
            "num_steps": torch.tensor([item["num_steps"] for item in batch]),
        }
    
    def save_cache(self, name: str):
        """Save processed data to cache"""
        cache_path = Path(self.config.cache_dir) / f"{name}.pkl"
        
        cache_data = {
            "preference_data": self.preference_dataset.data,
            "reasoning_data": self.reasoning_dataset.data,
            "config": self.config
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved cache to {cache_path}")
    
    def load_cache(self, name: str) -> bool:
        """Load processed data from cache"""
        cache_path = Path(self.config.cache_dir) / f"{name}.pkl"
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.preference_dataset.data = cache_data["preference_data"]
            self.reasoning_dataset.data = cache_data["reasoning_data"]
            
            logger.info(f"Loaded cache from {cache_path}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False