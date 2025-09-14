"""
Multi-Reward Model System
Implements separate reward models for different aspects of quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from transformers import AutoModel, AutoTokenizer
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MultiRewardConfig:
    """Configuration for multi-reward models"""
    # Model paths
    base_model_name: str = "microsoft/deberta-v3-base"
    
    # Reward weights
    factual_weight: float = 2.0
    coherence_weight: float = 1.5
    helpfulness_weight: float = 1.0
    safety_weight: float = 2.5
    
    # Training settings
    hidden_size: int = 768
    num_labels: int = 1
    dropout_rate: float = 0.1
    
    # Factual accuracy settings
    fact_check_temperature: float = 0.3
    min_fact_confidence: float = 0.7
    
    # Coherence settings
    coherence_window_size: int = 3
    coherence_threshold: float = 0.6
    
    # Safety settings
    safety_categories: List[str] = None
    
    def __post_init__(self):
        if self.safety_categories is None:
            self.safety_categories = [
                "violence", "hate", "sexual", "self-harm",
                "illegal", "deception", "privacy", "medical"
            ]

class RewardHead(nn.Module):
    """Base reward prediction head"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 4, config.num_labels)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool hidden states
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        
        return self.classifier(pooled)

class FactualAccuracyReward(nn.Module):
    """Reward model for factual accuracy"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained model for fact embeddings
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        
        # Fact verification head
        self.fact_head = RewardHead(config)
        
        # Claim extraction patterns
        self.claim_patterns = [
            r"(?:It is|This is|That is|They are|We are|I am|You are)\s+(.+?)(?:\.|,|;|$)",
            r"(?:The|A|An)\s+(.+?)\s+(?:is|are|was|were|has|have|had)\s+(.+?)(?:\.|,|;|$)",
            r"(?:According to|Studies show|Research indicates|Data suggests)\s+(.+?)(?:\.|,|;|$)",
        ]
        
        # Knowledge base interface (simplified)
        self.knowledge_cache = {}
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains factual claims
            for pattern in self.claim_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                claims.extend([m if isinstance(m, str) else " ".join(m) for m in matches])
            
            # Also include full sentences that look factual
            if any(word in sentence.lower() for word in ["is", "are", "was", "were", "has", "have"]):
                claims.append(sentence)
        
        return list(set(claims))  # Remove duplicates
    
    def verify_claim(self, claim: str, context: Optional[str] = None) -> float:
        """Verify a single claim (simplified version)"""
        # In practice, this would query a knowledge base or use retrieval
        
        # Encode claim
        inputs = self.tokenizer(
            claim,
            context or "",
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            claim_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Get fact score from specialized head
            fact_score = torch.sigmoid(self.fact_head(claim_embedding))
        
        return fact_score.item()
    
    def forward(
        self,
        text: str,
        context: Optional[str] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """Compute factual accuracy reward"""
        claims = self.extract_claims(text)
        
        if not claims:
            return 0.5 if not return_details else {"score": 0.5, "claims": [], "verifications": {}}
        
        verifications = {}
        scores = []
        
        for claim in claims:
            score = self.verify_claim(claim, context)
            verifications[claim] = score
            scores.append(score)
        
        # Aggregate scores
        avg_score = np.mean(scores)
        
        # Penalize if any claim is very low confidence
        min_score = min(scores)
        if min_score < self.config.min_fact_confidence:
            penalty = (self.config.min_fact_confidence - min_score) * 0.5
            avg_score = max(0, avg_score - penalty)
        
        if return_details:
            return {
                "score": avg_score,
                "claims": claims,
                "verifications": verifications,
                "min_score": min_score,
                "num_claims": len(claims)
            }
        
        return avg_score

class CoherenceReward(nn.Module):
    """Reward model for logical coherence and consistency"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        # Coherence analysis model
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        self.coherence_head = RewardHead(config)
        
        # Sentence-level coherence scorer
        self.sentence_coherence = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Contradiction detection patterns
        self.contradiction_patterns = [
            (r"not\s+(.+)", r"(?:is|are|was|were)\s+\1"),
            (r"never\s+(.+)", r"always\s+\1"),
            (r"all\s+(.+)", r"no\s+\1"),
            (r"increase", r"decrease"),
            (r"higher", r"lower"),
        ]
    
    def check_contradictions(self, sentences: List[str]) -> List[Tuple[int, int, str]]:
        """Check for contradictions between sentences"""
        contradictions = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sent1, sent2 = sentences[i].lower(), sentences[j].lower()
                
                # Check contradiction patterns
                for pattern1, pattern2 in self.contradiction_patterns:
                    if re.search(pattern1, sent1) and re.search(pattern2, sent2):
                        contradictions.append((i, j, "pattern_match"))
                    elif re.search(pattern2, sent1) and re.search(pattern1, sent2):
                        contradictions.append((i, j, "pattern_match"))
        
        return contradictions
    
    def compute_sentence_coherence(
        self,
        sentences: List[str]
    ) -> List[float]:
        """Compute coherence scores between adjacent sentences"""
        if len(sentences) < 2:
            return [1.0]
        
        coherence_scores = []
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        
        for i in range(len(sentences) - 1):
            # Encode adjacent sentences
            inputs = tokenizer(
                sentences[i],
                sentences[i + 1],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                
                # Get embeddings for both sentences
                hidden = outputs.last_hidden_state
                sent1_emb = hidden[:, :len(sentences[i].split()), :].mean(dim=1)
                sent2_emb = hidden[:, len(sentences[i].split()):, :].mean(dim=1)
                
                # Compute coherence score
                combined = torch.cat([sent1_emb, sent2_emb], dim=-1)
                coherence = self.sentence_coherence(combined)
                coherence_scores.append(coherence.item())
        
        return coherence_scores
    
    def forward(
        self,
        text: str,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """Compute coherence reward"""
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 1.0 if not return_details else {"score": 1.0, "details": "Too short to assess"}
        
        # Check contradictions
        contradictions = self.check_contradictions(sentences)
        
        # Compute sentence-level coherence
        coherence_scores = self.compute_sentence_coherence(sentences)
        
        # Analyze topic flow
        window_scores = []
        for i in range(len(sentences) - self.config.coherence_window_size + 1):
            window = sentences[i:i + self.config.coherence_window_size]
            window_text = " ".join(window)
            
            # Encode window
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            inputs = tokenizer(
                window_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                window_embedding = outputs.last_hidden_state.mean(dim=1)
                window_score = torch.sigmoid(self.coherence_head(window_embedding))
                window_scores.append(window_score.item())
        
        # Aggregate scores
        base_score = np.mean(coherence_scores) if coherence_scores else 1.0
        window_score = np.mean(window_scores) if window_scores else 1.0
        
        # Apply penalties
        contradiction_penalty = len(contradictions) * 0.2
        low_coherence_penalty = sum(1 for s in coherence_scores if s < self.config.coherence_threshold) * 0.1
        
        final_score = max(0, min(1, base_score * window_score - contradiction_penalty - low_coherence_penalty))
        
        if return_details:
            return {
                "score": final_score,
                "sentence_coherence": coherence_scores,
                "window_scores": window_scores,
                "contradictions": contradictions,
                "num_sentences": len(sentences)
            }
        
        return final_score

class HelpfulnessReward(nn.Module):
    """Reward model for helpfulness and informativeness"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        self.helpfulness_head = RewardHead(config)
        
        # Specific helpfulness indicators
        self.helpfulness_patterns = {
            "instructive": [
                r"(?:you can|you should|try to|make sure to|remember to)",
                r"(?:first|second|then|next|finally)",
                r"(?:for example|such as|including|like)",
            ],
            "informative": [
                r"(?:because|since|as|due to)",
                r"(?:this means|in other words|that is)",
                r"(?:specifically|particularly|especially)",
            ],
            "actionable": [
                r"(?:step \d+|follow these steps|here's how)",
                r"(?:to do this|to achieve|to accomplish)",
                r"(?:click|select|choose|enter|type)",
            ],
        }
    
    def extract_helpful_features(self, text: str) -> Dict[str, float]:
        """Extract features indicating helpfulness"""
        features = defaultdict(float)
        
        # Check for helpful patterns
        for category, patterns in self.helpfulness_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            features[f"pattern_{category}"] = min(count / 10.0, 1.0)  # Normalize
        
        # Information density (unique words / total words)
        words = text.lower().split()
        if words:
            features["info_density"] = len(set(words)) / len(words)
        
        # Presence of examples
        features["has_examples"] = float(bool(re.search(r"(?:for example|e\.g\.|such as)", text, re.IGNORECASE)))
        
        # Presence of structure
        features["has_structure"] = float(bool(re.search(r"(?:\d+\.|•|→|first|second|third)", text)))
        
        # Length appropriateness
        word_count = len(words)
        if 50 <= word_count <= 500:
            features["good_length"] = 1.0
        elif word_count < 50:
            features["good_length"] = word_count / 50.0
        else:
            features["good_length"] = max(0, 1.0 - (word_count - 500) / 1000.0)
        
        return dict(features)
    
    def forward(
        self,
        text: str,
        query: Optional[str] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """Compute helpfulness reward"""
        features = self.extract_helpful_features(text)
        
        # Encode text (and query if provided)
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        
        if query:
            inputs = tokenizer(
                query,
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        else:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            text_embedding = outputs.last_hidden_state.mean(dim=1)
            base_score = torch.sigmoid(self.helpfulness_head(text_embedding)).item()
        
        # Combine with feature scores
        feature_score = np.mean(list(features.values()))
        final_score = 0.7 * base_score + 0.3 * feature_score
        
        # Boost score if query is well-addressed
        if query:
            # Simple relevance check
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words) / len(query_words) if query_words else 0
            final_score = final_score * (0.8 + 0.2 * overlap)
        
        if return_details:
            return {
                "score": final_score,
                "base_score": base_score,
                "features": features,
                "feature_score": feature_score,
                "query_addressed": overlap if query else None
            }
        
        return final_score

class SafetyReward(nn.Module):
    """Reward model for safety and harm prevention"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        
        # Multiple safety heads for different categories
        self.safety_heads = nn.ModuleDict({
            category: RewardHead(config)
            for category in config.safety_categories
        })
        
        # Overall safety aggregator
        self.safety_aggregator = nn.Sequential(
            nn.Linear(len(config.safety_categories), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Unsafe patterns
        self.unsafe_patterns = {
            "violence": [
                r"(?:kill|murder|assault|attack|hurt|harm|weapon|gun|knife|bomb)",
                r"(?:fight|punch|shoot|stab|poison|torture)",
            ],
            "hate": [
                r"(?:hate|discriminate|racist|sexist|bigot)",
                r"(?:inferior|superior race|ethnic slur)",
            ],
            "self-harm": [
                r"(?:suicide|self-harm|cut yourself|end your life)",
                r"(?:worthless|better off dead|no point living)",
            ],
            "illegal": [
                r"(?:steal|theft|fraud|hack|pirate|counterfeit)",
                r"(?:drug dealing|trafficking|money laundering)",
            ],
            "deception": [
                r"(?:lie|deceive|trick|scam|phishing|impersonate)",
                r"(?:fake identity|false information|misinformation)",
            ],
            "privacy": [
                r"(?:personal information|private data|password|ssn|credit card)",
                r"(?:dox|leak|expose personal|share private)",
            ],
        }
    
    def detect_unsafe_content(self, text: str) -> Dict[str, float]:
        """Detect unsafe content by category"""
        unsafe_scores = {}
        
        for category, patterns in self.unsafe_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score = max(score, min(matches / 3.0, 1.0))  # Cap at 1.0
            unsafe_scores[category] = score
        
        return unsafe_scores
    
    def forward(
        self,
        text: str,
        context: Optional[str] = None,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """Compute safety reward (higher is safer)"""
        # Quick pattern-based detection
        pattern_scores = self.detect_unsafe_content(text)
        
        # Deep analysis with model
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
        inputs = tokenizer(
            text,
            context or "",
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            text_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Get safety scores for each category
            category_scores = {}
            for category, head in self.safety_heads.items():
                score = torch.sigmoid(head(text_embedding)).item()
                # Combine with pattern score (pattern detection overrides if high)
                pattern_penalty = pattern_scores.get(category, 0.0)
                category_scores[category] = max(0, score - pattern_penalty)
            
            # Aggregate safety score
            all_scores = torch.tensor(list(category_scores.values())).unsqueeze(0)
            overall_safety = self.safety_aggregator(all_scores).item()
        
        # Apply strict penalties for severe violations
        min_category_score = min(category_scores.values())
        if min_category_score < 0.3:  # Severe safety issue
            overall_safety *= 0.5
        
        if return_details:
            return {
                "score": overall_safety,
                "category_scores": category_scores,
                "pattern_detections": pattern_scores,
                "min_category": min(category_scores.items(), key=lambda x: x[1]),
                "flagged_categories": [cat for cat, score in category_scores.items() if score < 0.5]
            }
        
        return overall_safety

class RewardAggregator(nn.Module):
    """Aggregates multiple reward signals with learnable weights"""
    def __init__(self, config: MultiRewardConfig):
        super().__init__()
        self.config = config
        
        # Learnable aggregation weights
        self.reward_weights = nn.Parameter(torch.tensor([
            config.factual_weight,
            config.coherence_weight,
            config.helpfulness_weight,
            config.safety_weight
        ]))
        
        # Non-linear aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    
    def forward(
        self,
        rewards: Dict[str, float],
        return_weighted: bool = True
    ) -> Union[float, Dict[str, float]]:
        """Aggregate multiple rewards into final score"""
        # Extract individual rewards
        factual = rewards.get("factual", 0.5)
        coherence = rewards.get("coherence", 0.5)
        helpfulness = rewards.get("helpfulness", 0.5)
        safety = rewards.get("safety", 1.0)
        
        reward_tensor = torch.tensor([factual, coherence, helpfulness, safety])
        
        if return_weighted:
            # Weighted sum with learnable weights
            weights = F.softmax(self.reward_weights, dim=0)
            weighted_rewards = reward_tensor * weights
            
            # Non-linear aggregation
            final_score = self.aggregator(weighted_rewards.unsqueeze(0)).squeeze()
            
            return {
                "final_score": final_score.item(),
                "weighted_rewards": {
                    "factual": (factual * weights[0]).item(),
                    "coherence": (coherence * weights[1]).item(),
                    "helpfulness": (helpfulness * weights[2]).item(),
                    "safety": (safety * weights[3]).item()
                },
                "weights": {
                    "factual": weights[0].item(),
                    "coherence": weights[1].item(),
                    "helpfulness": weights[2].item(),
                    "safety": weights[3].item()
                }
            }
        else:
            # Simple average
            return reward_tensor.mean().item()

class MultiRewardModel:
    """Unified multi-reward model system"""
    def __init__(self, config: MultiRewardConfig):
        self.config = config
        
        # Initialize individual reward models
        self.factual_model = FactualAccuracyReward(config)
        self.coherence_model = CoherenceReward(config)
        self.helpfulness_model = HelpfulnessReward(config)
        self.safety_model = SafetyReward(config)
        self.aggregator = RewardAggregator(config)
        
        # Move to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def to(self, device):
        """Move all models to device"""
        self.factual_model.to(device)
        self.coherence_model.to(device)
        self.helpfulness_model.to(device)
        self.safety_model.to(device)
        self.aggregator.to(device)
        return self
    
    def compute_rewards(
        self,
        text: str,
        context: Optional[str] = None,
        query: Optional[str] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """Compute all reward components"""
        rewards = {}
        details = {}
        
        # Factual accuracy
        factual_result = self.factual_model(text, context, return_details=return_details)
        if return_details:
            rewards["factual"] = factual_result["score"]
            details["factual"] = factual_result
        else:
            rewards["factual"] = factual_result
        
        # Coherence
        coherence_result = self.coherence_model(text, return_details=return_details)
        if return_details:
            rewards["coherence"] = coherence_result["score"]
            details["coherence"] = coherence_result
        else:
            rewards["coherence"] = coherence_result
        
        # Helpfulness
        helpfulness_result = self.helpfulness_model(text, query, return_details=return_details)
        if return_details:
            rewards["helpfulness"] = helpfulness_result["score"]
            details["helpfulness"] = helpfulness_result
        else:
            rewards["helpfulness"] = helpfulness_result
        
        # Safety
        safety_result = self.safety_model(text, context, return_details=return_details)
        if return_details:
            rewards["safety"] = safety_result["score"]
            details["safety"] = safety_result
        else:
            rewards["safety"] = safety_result
        
        # Aggregate
        aggregated = self.aggregator(rewards, return_weighted=True)
        
        result = {
            "rewards": rewards,
            "aggregated": aggregated,
            "final_score": aggregated["final_score"]
        }
        
        if return_details:
            result["details"] = details
        
        return result
    
    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step for reward models"""
        losses = {}
        
        # Each reward model can be trained separately or jointly
        # This is a simplified version
        
        texts = batch["text"]
        labels = batch["labels"]  # Assume dict of reward labels
        
        total_loss = 0.0
        
        # Train each component
        for reward_type, model in [
            ("factual", self.factual_model),
            ("coherence", self.coherence_model),
            ("helpfulness", self.helpfulness_model),
            ("safety", self.safety_model)
        ]:
            if reward_type in labels:
                predictions = []
                targets = []
                
                for i, text in enumerate(texts):
                    pred = model(text)
                    predictions.append(pred)
                    targets.append(labels[reward_type][i])
                
                pred_tensor = torch.tensor(predictions, requires_grad=True)
                target_tensor = torch.tensor(targets)
                
                loss = F.mse_loss(pred_tensor, target_tensor)
                losses[f"loss/{reward_type}"] = loss.item()
                total_loss += loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses["loss/total"] = total_loss.item()
        
        return losses