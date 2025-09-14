"""
Self-Evaluation and Correction System
Implements automated fact-checking, logical consistency checking, and self-correction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from dataclasses import dataclass, field
import logging
import re
from collections import defaultdict
import json
from pathlib import Path
import requests
from urllib.parse import quote
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class SelfEvaluationConfig:
    """Configuration for self-evaluation system"""
    # Fact checking
    fact_check_sources: List[str] = field(default_factory=lambda: [
        "wikipedia", "wikidata", "knowledge_graph"
    ])
    fact_confidence_threshold: float = 0.7
    max_fact_checks_per_claim: int = 3
    
    # Logical consistency
    logic_rules_path: Optional[str] = None
    consistency_threshold: float = 0.8
    max_reasoning_depth: int = 5
    
    # Confidence scoring
    confidence_calibration_bins: int = 10
    confidence_temperature: float = 1.0
    
    # Self-correction
    max_correction_iterations: int = 3
    correction_threshold: float = 0.6
    correction_temperature: float = 0.7
    
    # Model settings
    hidden_size: int = 768
    dropout_rate: float = 0.1

class FactChecker(nn.Module):
    """Automated fact-checking against knowledge bases"""
    def __init__(self, config: SelfEvaluationConfig):
        super().__init__()
        self.config = config
        
        # Claim decomposition network
        self.claim_decomposer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Fact verification head
        self.verification_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 3)  # Supported, Refuted, Not Enough Info
        )
        
        # Knowledge retrieval cache
        self.knowledge_cache = {}
        
        # Setup knowledge sources
        self.knowledge_sources = self._setup_knowledge_sources()
    
    def _setup_knowledge_sources(self) -> Dict[str, Any]:
        """Setup connections to knowledge sources"""
        sources = {}
        
        if "wikipedia" in self.config.fact_check_sources:
            sources["wikipedia"] = {
                "endpoint": "https://en.wikipedia.org/w/api.php",
                "search_params": {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "utf8": 1
                }
            }
        
        if "wikidata" in self.config.fact_check_sources:
            sources["wikidata"] = {
                "endpoint": "https://www.wikidata.org/w/api.php",
                "sparql_endpoint": "https://query.wikidata.org/sparql"
            }
        
        return sources
    
    def decompose_claim(self, claim: str, hidden_states: torch.Tensor) -> List[Dict[str, Any]]:
        """Decompose complex claim into atomic facts"""
        # Parse claim structure
        atomic_facts = []
        
        # Extract entities and relations
        entities = self._extract_entities(claim)
        relations = self._extract_relations(claim)
        
        # Simple decomposition based on conjunctions
        sub_claims = re.split(r'\s+(?:and|or|but)\s+', claim, flags=re.IGNORECASE)
        
        for sub_claim in sub_claims:
            # Further decompose based on structure
            if " because " in sub_claim.lower():
                parts = sub_claim.split(" because ", 1)
                atomic_facts.append({
                    "claim": parts[0].strip(),
                    "type": "statement",
                    "reason": parts[1].strip()
                })
            elif " if " in sub_claim.lower():
                parts = re.split(r'\s+if\s+', sub_claim, flags=re.IGNORECASE)
                atomic_facts.append({
                    "claim": parts[0].strip(),
                    "type": "conditional",
                    "condition": parts[1].strip() if len(parts) > 1 else None
                })
            else:
                atomic_facts.append({
                    "claim": sub_claim.strip(),
                    "type": "simple",
                    "entities": [e for e in entities if e in sub_claim],
                    "relations": [r for r in relations if r in sub_claim]
                })
        
        return atomic_facts
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simplified entity extraction
        entities = []
        
        # Capitalized words (likely proper nouns)
        entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        
        # Numbers and dates
        entities.extend(re.findall(r'\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)\b', text))
        
        return list(set(entities))
    
    def _extract_relations(self, text: str) -> List[str]:
        """Extract relations/predicates from text"""
        relations = []
        
        # Common relation patterns
        relation_patterns = [
            r'\b(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)',
            r'\b(?:has|have|had)\s+(\w+)',
            r'\b(\w+ed)\s+(?:by|in|at|on)',
            r'\b(?:located|found|discovered)\s+(?:in|at)',
        ]
        
        for pattern in relation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relations.extend(matches)
        
        return list(set(relations))
    
    async def verify_fact_async(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously verify a single fact against knowledge sources"""
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for source_name, source_config in self.knowledge_sources.items():
                if source_name == "wikipedia":
                    task = self._check_wikipedia_async(session, fact, source_config)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        verification_scores = []
        evidence = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Fact checking error: {result}")
                continue
            
            if result:
                verification_scores.append(result["score"])
                evidence.append(result["evidence"])
        
        if verification_scores:
            avg_score = np.mean(verification_scores)
            confidence = self._compute_verification_confidence(verification_scores)
            
            return {
                "fact": fact,
                "verification_score": avg_score,
                "confidence": confidence,
                "evidence": evidence,
                "sources_checked": len(verification_scores)
            }
        else:
            return {
                "fact": fact,
                "verification_score": 0.5,  # Unknown
                "confidence": 0.0,
                "evidence": [],
                "sources_checked": 0
            }
    
    async def _check_wikipedia_async(
        self,
        session: aiohttp.ClientSession,
        fact: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check fact against Wikipedia"""
        try:
            # Search for relevant articles
            search_query = " ".join(fact.get("entities", [])) + " " + fact["claim"]
            params = {
                **config["search_params"],
                "srsearch": search_query,
                "srlimit": 3
            }
            
            async with session.get(config["endpoint"], params=params) as response:
                data = await response.json()
                
                if "query" in data and "search" in data["query"]:
                    articles = data["query"]["search"]
                    
                    if articles:
                        # Simple relevance scoring based on title and snippet
                        relevance_scores = []
                        for article in articles:
                            title_score = self._text_similarity(fact["claim"], article["title"])
                            snippet_score = self._text_similarity(fact["claim"], article["snippet"])
                            relevance_scores.append(max(title_score, snippet_score))
                        
                        best_score = max(relevance_scores)
                        best_article = articles[relevance_scores.index(best_score)]
                        
                        return {
                            "score": best_score,
                            "evidence": {
                                "source": "wikipedia",
                                "title": best_article["title"],
                                "snippet": best_article["snippet"],
                                "relevance": best_score
                            }
                        }
        
        except Exception as e:
            logger.error(f"Wikipedia fact check error: {e}")
        
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity"""
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _compute_verification_confidence(self, scores: List[float]) -> float:
        """Compute confidence in verification result"""
        if not scores:
            return 0.0
        
        # High confidence if scores agree
        variance = np.var(scores)
        mean_score = np.mean(scores)
        
        # Confidence based on agreement and extremity
        agreement_confidence = 1.0 / (1.0 + variance)
        extremity_confidence = abs(mean_score - 0.5) * 2  # How far from uncertain
        
        return (agreement_confidence + extremity_confidence) / 2
    
    def forward(
        self,
        text: str,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Perform fact checking on text"""
        # Extract claims
        claims = self._extract_claims(text)
        
        # Decompose complex claims
        all_facts = []
        for claim in claims:
            if hidden_states is not None:
                facts = self.decompose_claim(claim, hidden_states)
            else:
                facts = [{"claim": claim, "type": "simple"}]
            all_facts.extend(facts)
        
        # Verify facts asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        verification_results = []
        for fact in all_facts:
            result = loop.run_until_complete(self.verify_fact_async(fact))
            verification_results.append(result)
        
        loop.close()
        
        # Aggregate results
        verified_facts = [r for r in verification_results if r["confidence"] > self.config.fact_confidence_threshold]
        unverified_facts = [r for r in verification_results if r["confidence"] <= self.config.fact_confidence_threshold]
        
        overall_score = np.mean([r["verification_score"] for r in verification_results]) if verification_results else 0.5
        
        return {
            "overall_score": overall_score,
            "num_facts": len(all_facts),
            "verified_facts": verified_facts,
            "unverified_facts": unverified_facts,
            "fact_details": verification_results
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains factual claims
            if any(pattern in sentence.lower() for pattern in [
                "is", "are", "was", "were", "has", "have", "had",
                "will", "would", "should", "must", "can", "could"
            ]):
                claims.append(sentence)
        
        return claims

class LogicalConsistencyChecker(nn.Module):
    """Checks for logical consistency in generated text"""
    def __init__(self, config: SelfEvaluationConfig):
        super().__init__()
        self.config = config
        
        # Logical relation encoder
        self.relation_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Consistency scorer
        self.consistency_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Logical rules
        self.logical_rules = self._load_logical_rules()
        
        # Inference cache
        self.inference_cache = {}
    
    def _load_logical_rules(self) -> Dict[str, Any]:
        """Load logical rules for consistency checking"""
        rules = {
            "contradictions": [
                # Negation patterns
                (r"(?:all|every)\s+(\w+)", r"(?:no|none)\s+\1"),
                (r"always\s+(\w+)", r"never\s+\1"),
                (r"(\w+)\s+(?:increases?|grows?)", r"\1\s+(?:decreases?|shrinks?)"),
                
                # Logical opposites
                (r"if\s+(.+?)\s+then\s+(.+)", r"if\s+\1\s+then\s+not\s+\2"),
                (r"(\w+)\s+causes?\s+(\w+)", r"\1\s+prevents?\s+\2"),
            ],
            "implications": [
                # If-then rules
                (r"all\s+(\w+)\s+are\s+(\w+)", r"if\s+.*\s+is\s+(?:a|an)\s+\1.*then.*\2"),
                (r"(\w+)\s+requires?\s+(\w+)", r"if\s+.*\1.*then.*must.*\2"),
            ],
            "transitivity": [
                # A->B, B->C implies A->C
                (r"(\w+)\s+(?:is|are)\s+(\w+)", r"\2\s+(?:is|are)\s+(\w+)", r"\1\s+(?:is|are)\s+\3"),
            ]
        }
        
        # Load custom rules if provided
        if self.config.logic_rules_path:
            try:
                with open(self.config.logic_rules_path, 'r') as f:
                    custom_rules = json.load(f)
                    rules.update(custom_rules)
            except Exception as e:
                logger.warning(f"Failed to load custom logic rules: {e}")
        
        return rules
    
    def extract_logical_statements(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical statements and their relationships"""
        statements = []
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract logical structure
            statement = {
                "id": i,
                "text": sentence,
                "type": "simple",
                "components": []
            }
            
            # Check for conditional statements
            if_match = re.search(r"if\s+(.+?)\s+then\s+(.+)", sentence, re.IGNORECASE)
            if if_match:
                statement["type"] = "conditional"
                statement["condition"] = if_match.group(1)
                statement["consequence"] = if_match.group(2)
            
            # Check for causal statements
            cause_match = re.search(r"(.+?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(.+)", sentence, re.IGNORECASE)
            if cause_match:
                statement["type"] = "causal"
                statement["cause"] = cause_match.group(1)
                statement["effect"] = cause_match.group(2)
            
            # Check for universal statements
            universal_match = re.search(r"(?:all|every|no|none)\s+(.+?)\s+(?:is|are|has|have)\s+(.+)", sentence, re.IGNORECASE)
            if universal_match:
                statement["type"] = "universal"
                statement["subject"] = universal_match.group(1)
                statement["predicate"] = universal_match.group(2)
            
            statements.append(statement)
        
        return statements
    
    def check_contradiction(self, stmt1: Dict[str, Any], stmt2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if two statements contradict each other"""
        text1 = stmt1["text"].lower()
        text2 = stmt2["text"].lower()
        
        # Check contradiction patterns
        for pattern1, pattern2 in self.logical_rules["contradictions"]:
            if re.search(pattern1, text1) and re.search(pattern2, text2):
                return {
                    "type": "direct_contradiction",
                    "statement1": stmt1,
                    "statement2": stmt2,
                    "pattern": (pattern1, pattern2)
                }
            if re.search(pattern2, text1) and re.search(pattern1, text2):
                return {
                    "type": "direct_contradiction",
                    "statement1": stmt1,
                    "statement2": stmt2,
                    "pattern": (pattern2, pattern1)
                }
        
        # Check logical contradictions
        if stmt1["type"] == "conditional" and stmt2["type"] == "conditional":
            if stmt1.get("condition") == stmt2.get("condition"):
                # Same condition, opposite consequences
                if self._are_opposites(stmt1.get("consequence", ""), stmt2.get("consequence", "")):
                    return {
                        "type": "conditional_contradiction",
                        "statement1": stmt1,
                        "statement2": stmt2,
                        "conflict": "opposite_consequences"
                    }
        
        return None
    
    def _are_opposites(self, text1: str, text2: str) -> bool:
        """Check if two texts express opposite meanings"""
        # Simple negation check
        if "not" in text1 and "not" not in text2:
            return text1.replace("not ", "") == text2
        if "not" in text2 and "not" not in text1:
            return text2.replace("not ", "") == text1
        
        # Antonym patterns
        antonym_pairs = [
            ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
            ("positive", "negative"), ("true", "false"), ("yes", "no")
        ]
        
        for word1, word2 in antonym_pairs:
            if word1 in text1 and word2 in text2:
                return True
            if word2 in text1 and word1 in text2:
                return True
        
        return False
    
    def check_logical_consistency(
        self,
        statements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check logical consistency across statements"""
        contradictions = []
        inconsistencies = []
        
        # Pairwise contradiction checking
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                contradiction = self.check_contradiction(statements[i], statements[j])
                if contradiction:
                    contradictions.append(contradiction)
        
        # Check transitive consistency
        implications = self._extract_implications(statements)
        transitive_violations = self._check_transitivity(implications)
        
        # Compute consistency score
        total_pairs = len(statements) * (len(statements) - 1) // 2
        if total_pairs > 0:
            consistency_score = 1.0 - (len(contradictions) + len(transitive_violations)) / total_pairs
        else:
            consistency_score = 1.0
        
        return {
            "consistency_score": consistency_score,
            "contradictions": contradictions,
            "transitive_violations": transitive_violations,
            "num_statements": len(statements),
            "is_consistent": consistency_score >= self.config.consistency_threshold
        }
    
    def _extract_implications(self, statements: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Extract implication relationships"""
        implications = []
        
        for stmt in statements:
            if stmt["type"] == "conditional":
                implications.append((stmt.get("condition", ""), stmt.get("consequence", "")))
            elif stmt["type"] == "causal":
                implications.append((stmt.get("cause", ""), stmt.get("effect", "")))
        
        return implications
    
    def _check_transitivity(self, implications: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Check transitivity violations"""
        violations = []
        
        # Build implication graph
        graph = defaultdict(set)
        for a, b in implications:
            graph[a].add(b)
        
        # Check transitivity
        for a in graph:
            for b in graph[a]:
                for c in graph.get(b, set()):
                    # A->B and B->C, check if A->C exists or contradicted
                    if c not in graph[a]:
                        # Check if opposite exists
                        for d in graph[a]:
                            if self._are_opposites(c, d):
                                violations.append({
                                    "type": "transitivity_violation",
                                    "chain": [a, b, c],
                                    "contradiction": d
                                })
        
        return violations
    
    def forward(
        self,
        text: str,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Check logical consistency of text"""
        # Extract logical statements
        statements = self.extract_logical_statements(text)
        
        # Check consistency
        consistency_result = self.check_logical_consistency(statements)
        
        # Use neural model for additional scoring if hidden states provided
        if hidden_states is not None and len(statements) > 1:
            # Encode statement pairs
            pair_scores = []
            
            for i in range(len(statements) - 1):
                # Simple concatenation of adjacent statements
                pair_hidden = hidden_states[i:i+2].flatten().unsqueeze(0)
                
                # Pad or truncate to expected size
                expected_size = self.config.hidden_size * 2
                if pair_hidden.size(1) < expected_size:
                    pair_hidden = F.pad(pair_hidden, (0, expected_size - pair_hidden.size(1)))
                elif pair_hidden.size(1) > expected_size:
                    pair_hidden = pair_hidden[:, :expected_size]
                
                score = self.consistency_head(pair_hidden)
                pair_scores.append(score.item())
            
            neural_consistency = np.mean(pair_scores) if pair_scores else 1.0
            
            # Combine with rule-based score
            consistency_result["neural_score"] = neural_consistency
            consistency_result["combined_score"] = (
                0.7 * consistency_result["consistency_score"] +
                0.3 * neural_consistency
            )
        
        return consistency_result

class ConfidenceScorer(nn.Module):
    """Scores model confidence and calibrates predictions"""
    def __init__(self, config: SelfEvaluationConfig):
        super().__init__()
        self.config = config
        
        # Confidence prediction network
        self.confidence_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation heads
        self.epistemic_head = nn.Linear(config.hidden_size, 1)  # Model uncertainty
        self.aleatoric_head = nn.Linear(config.hidden_size, 1)  # Data uncertainty
        
        # Calibration parameters
        self.temperature = nn.Parameter(torch.ones(1) * config.confidence_temperature)
        self.calibration_bins = torch.linspace(0, 1, config.confidence_calibration_bins + 1)
        
        # Calibration statistics
        self.register_buffer("bin_accuracies", torch.zeros(config.confidence_calibration_bins))
        self.register_buffer("bin_counts", torch.zeros(config.confidence_calibration_bins))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute confidence scores"""
        # Pool hidden states if needed
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        
        # Base confidence
        confidence = self.confidence_net(pooled)
        
        # Uncertainty estimates
        epistemic_uncertainty = torch.sigmoid(self.epistemic_head(pooled))
        aleatoric_uncertainty = torch.sigmoid(self.aleatoric_head(pooled))
        
        # Temperature scaling for calibration
        if logits is not None:
            calibrated_logits = logits / self.temperature
            calibrated_probs = F.softmax(calibrated_logits, dim=-1)
            max_prob = calibrated_probs.max(dim=-1)[0]
            
            # Adjust confidence based on calibrated probability
            confidence = 0.7 * confidence + 0.3 * max_prob.unsqueeze(-1)
        
        # Total uncertainty
        total_uncertainty = torch.sqrt(
            epistemic_uncertainty ** 2 + aleatoric_uncertainty ** 2
        )
        
        # Adjusted confidence (accounting for uncertainty)
        adjusted_confidence = confidence * (1 - total_uncertainty)
        
        return {
            "confidence": confidence.squeeze(-1),
            "adjusted_confidence": adjusted_confidence.squeeze(-1),
            "epistemic_uncertainty": epistemic_uncertainty.squeeze(-1),
            "aleatoric_uncertainty": aleatoric_uncertainty.squeeze(-1),
            "total_uncertainty": total_uncertainty.squeeze(-1)
        }
    
    def update_calibration_stats(
        self,
        confidences: torch.Tensor,
        correctness: torch.Tensor
    ):
        """Update calibration statistics"""
        with torch.no_grad():
            for i in range(self.config.confidence_calibration_bins):
                bin_mask = (
                    (confidences >= self.calibration_bins[i]) &
                    (confidences < self.calibration_bins[i + 1])
                )
                
                if bin_mask.sum() > 0:
                    self.bin_accuracies[i] = correctness[bin_mask].float().mean()
                    self.bin_counts[i] += bin_mask.sum()
    
    def compute_calibration_error(self) -> float:
        """Compute Expected Calibration Error (ECE)"""
        total_samples = self.bin_counts.sum()
        if total_samples == 0:
            return 0.0
        
        ece = 0.0
        for i in range(self.config.confidence_calibration_bins):
            if self.bin_counts[i] > 0:
                bin_confidence = (self.calibration_bins[i] + self.calibration_bins[i + 1]) / 2
                bin_accuracy = self.bin_accuracies[i]
                bin_weight = self.bin_counts[i] / total_samples
                
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece.item()

class SelfCorrectionPipeline(nn.Module):
    """Iterative self-correction based on evaluation results"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: SelfEvaluationConfig
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Correction strategy network
        self.correction_strategy = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 4)  # Rewrite, Refine, Expand, Simplify
        )
        
        # Error type classifier
        self.error_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 5)  # Factual, Logical, Clarity, Safety, Other
        )
    
    def identify_errors(
        self,
        text: str,
        evaluation_results: Dict[str, Any],
        hidden_states: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify specific errors to correct"""
        errors = []
        
        # Factual errors
        if "fact_check" in evaluation_results:
            for fact in evaluation_results["fact_check"]["unverified_facts"]:
                errors.append({
                    "type": "factual",
                    "severity": 1.0 - fact["confidence"],
                    "location": fact["fact"]["claim"],
                    "details": fact
                })
        
        # Logical errors
        if "consistency" in evaluation_results:
            for contradiction in evaluation_results["consistency"]["contradictions"]:
                errors.append({
                    "type": "logical",
                    "severity": 0.8,
                    "location": f"{contradiction['statement1']['text']} <-> {contradiction['statement2']['text']}",
                    "details": contradiction
                })
        
        # Low confidence regions
        if "confidence" in evaluation_results:
            conf = evaluation_results["confidence"]["adjusted_confidence"]
            if conf < self.config.correction_threshold:
                errors.append({
                    "type": "low_confidence",
                    "severity": 1.0 - conf,
                    "location": "overall",
                    "details": evaluation_results["confidence"]
                })
        
        # Use neural classifier for additional error detection
        if hidden_states is not None:
            pooled = hidden_states.mean(dim=1) if hidden_states.dim() == 3 else hidden_states
            error_logits = self.error_classifier(pooled)
            error_probs = F.softmax(error_logits, dim=-1)
            
            error_types = ["factual", "logical", "clarity", "safety", "other"]
            for i, (error_type, prob) in enumerate(zip(error_types, error_probs[0])):
                if prob > 0.5:
                    errors.append({
                        "type": error_type,
                        "severity": prob.item(),
                        "location": "neural_detected",
                        "details": {"probability": prob.item()}
                    })
        
        return errors
    
    def generate_correction_prompt(
        self,
        text: str,
        errors: List[Dict[str, Any]],
        iteration: int
    ) -> str:
        """Generate prompt for correction"""
        prompt = f"Please correct the following text (iteration {iteration + 1}):\n\n"
        prompt += f"Original text: {text}\n\n"
        prompt += "Identified issues:\n"
        
        for i, error in enumerate(errors[:5]):  # Limit to top 5 errors
            prompt += f"{i + 1}. {error['type'].title()} error"
            if error["severity"] > 0.7:
                prompt += " (HIGH PRIORITY)"
            prompt += f": {error['location'][:100]}...\n"
        
        prompt += "\nProvide a corrected version that addresses these issues while maintaining the original intent:"
        
        return prompt
    
    def apply_correction_strategy(
        self,
        text: str,
        strategy: str,
        error_details: Dict[str, Any]
    ) -> str:
        """Apply specific correction strategy"""
        if strategy == "rewrite":
            # Complete rewrite
            prompt = f"Completely rewrite the following text to fix {error_details['type']} errors:\n{text}"
        elif strategy == "refine":
            # Targeted refinement
            prompt = f"Refine the following text, specifically addressing {error_details['location']}:\n{text}"
        elif strategy == "expand":
            # Add clarification
            prompt = f"Expand the following text with clarifications for {error_details['type']} issues:\n{text}"
        elif strategy == "simplify":
            # Simplify for clarity
            prompt = f"Simplify the following text to improve clarity and fix {error_details['type']} issues:\n{text}"
        else:
            return text
        
        # Generate correction
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(text.split()) * 2,
                temperature=self.config.correction_temperature,
                do_sample=True,
            )
        
        corrected = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return corrected.strip()
    
    def forward(
        self,
        text: str,
        evaluation_results: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform iterative self-correction"""
        max_iterations = max_iterations or self.config.max_correction_iterations
        
        correction_history = []
        current_text = text
        
        for iteration in range(max_iterations):
            # Identify errors
            errors = self.identify_errors(current_text, evaluation_results, hidden_states)
            
            if not errors:
                break
            
            # Sort by severity
            errors.sort(key=lambda x: x["severity"], reverse=True)
            
            # Determine correction strategy
            if hidden_states is not None and errors:
                # Use neural strategy selection
                error_features = torch.zeros(self.config.hidden_size)
                for error in errors[:3]:  # Top 3 errors
                    error_features += torch.randn(self.config.hidden_size) * error["severity"]
                
                strategy_input = torch.cat([
                    hidden_states.mean(dim=1) if hidden_states.dim() == 3 else hidden_states,
                    error_features.unsqueeze(0)
                ], dim=-1)
                
                strategy_logits = self.correction_strategy(strategy_input)
                strategy_idx = strategy_logits.argmax(dim=-1).item()
                strategies = ["rewrite", "refine", "expand", "simplify"]
                strategy = strategies[strategy_idx]
            else:
                # Default strategy based on error type
                if errors[0]["type"] == "factual":
                    strategy = "rewrite"
                elif errors[0]["type"] == "logical":
                    strategy = "refine"
                else:
                    strategy = "expand"
            
            # Apply correction
            corrected_text = self.apply_correction_strategy(
                current_text,
                strategy,
                errors[0]
            )
            
            # Record correction
            correction_history.append({
                "iteration": iteration,
                "original": current_text,
                "corrected": corrected_text,
                "errors_addressed": errors[:3],  # Top 3
                "strategy": strategy
            })
            
            # Check if correction is better
            # (In practice, re-evaluate the corrected text)
            if len(corrected_text.strip()) > 10:  # Basic validity check
                current_text = corrected_text
            else:
                break
        
        return {
            "final_text": current_text,
            "num_iterations": len(correction_history),
            "correction_history": correction_history,
            "total_errors_found": sum(len(h["errors_addressed"]) for h in correction_history)
        }

class SelfEvaluationSystem:
    """Unified self-evaluation and correction system"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: SelfEvaluationConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize components
        self.fact_checker = FactChecker(config)
        self.consistency_checker = LogicalConsistencyChecker(config)
        self.confidence_scorer = ConfidenceScorer(config)
        self.correction_pipeline = SelfCorrectionPipeline(model, tokenizer, config)
        
        # Move to device
        device = next(model.parameters()).device
        self.fact_checker.to(device)
        self.consistency_checker.to(device)
        self.confidence_scorer.to(device)
        self.correction_pipeline.to(device)
    
    def evaluate(
        self,
        text: str,
        context: Optional[str] = None,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of generated text"""
        # Get model hidden states
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            logits = outputs.logits if hasattr(outputs, "logits") else None
        
        evaluation_results = {}
        
        # Fact checking
        fact_check_results = self.fact_checker(text, hidden_states)
        evaluation_results["fact_check"] = fact_check_results
        
        # Logical consistency
        consistency_results = self.consistency_checker(text, hidden_states)
        evaluation_results["consistency"] = consistency_results
        
        # Confidence scoring
        confidence_results = self.confidence_scorer(hidden_states, logits)
        evaluation_results["confidence"] = {
            k: v.item() if torch.is_tensor(v) and v.dim() == 0 else v
            for k, v in confidence_results.items()
        }
        
        # Overall quality score
        quality_score = self._compute_overall_quality(evaluation_results)
        evaluation_results["overall_quality"] = quality_score
        
        if not return_details:
            return {"quality_score": quality_score}
        
        return evaluation_results
    
    def evaluate_and_correct(
        self,
        text: str,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate and iteratively correct text"""
        # Initial evaluation
        evaluation_results = self.evaluate(text, context)
        
        # Get hidden states for correction
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        # Apply corrections if quality is low
        if evaluation_results["overall_quality"] < self.config.correction_threshold:
            correction_results = self.correction_pipeline(
                text,
                evaluation_results,
                hidden_states,
                max_iterations
            )
            
            # Re-evaluate corrected text
            final_evaluation = self.evaluate(correction_results["final_text"], context)
            
            return {
                "original_text": text,
                "corrected_text": correction_results["final_text"],
                "original_evaluation": evaluation_results,
                "final_evaluation": final_evaluation,
                "correction_details": correction_results,
                "quality_improvement": final_evaluation["overall_quality"] - evaluation_results["overall_quality"]
            }
        else:
            return {
                "original_text": text,
                "corrected_text": text,
                "evaluation": evaluation_results,
                "correction_needed": False
            }
    
    def _compute_overall_quality(self, evaluation_results: Dict[str, Any]) -> float:
        """Compute overall quality score from individual components"""
        scores = []
        weights = []
        
        # Fact checking score
        if "fact_check" in evaluation_results:
            scores.append(evaluation_results["fact_check"]["overall_score"])
            weights.append(2.0)  # High weight for factual accuracy
        
        # Consistency score
        if "consistency" in evaluation_results:
            score = evaluation_results["consistency"].get("combined_score",
                evaluation_results["consistency"]["consistency_score"])
            scores.append(score)
            weights.append(1.5)
        
        # Confidence score
        if "confidence" in evaluation_results:
            scores.append(evaluation_results["confidence"]["adjusted_confidence"])
            weights.append(1.0)
        
        if not scores:
            return 0.5
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight