"""
Advanced Reasoning Modules
Implements specialized reasoning heads for different domains with verification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
import re
from enum import Enum
import sympy
from collections import deque

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning"""
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    CODE = "code"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

@dataclass
class ReasoningConfig:
    """Configuration for reasoning modules"""
    hidden_size: int = 768
    num_reasoning_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.1
    max_reasoning_steps: int = 20
    step_embedding_size: int = 128
    
    # Verification settings
    verify_intermediate_steps: bool = True
    min_step_confidence: float = 0.6
    backtrack_on_error: bool = True
    
    # Mathematical reasoning
    use_symbolic_math: bool = True
    numerical_tolerance: float = 1e-6
    
    # Code reasoning
    max_execution_time: float = 5.0
    sandbox_execution: bool = True
    
    # Scientific reasoning
    use_knowledge_base: bool = True
    fact_verification: bool = True

class ReasoningModule(nn.Module):
    """Base reasoning module with step-by-step generation and verification"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Step encoder
        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=config.num_reasoning_layers
        )
        
        # Step generator
        self.step_generator = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=2,
            dropout=config.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Step type predictor
        self.step_type_head = nn.Linear(config.hidden_size * 2, len(ReasoningType))
        
        # Step validity predictor
        self.validity_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Step embeddings
        self.step_position_embeddings = nn.Embedding(
            config.max_reasoning_steps,
            config.step_embedding_size
        )
    
    def encode_context(
        self,
        input_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input context for reasoning"""
        # Add reasoning-specific encoding
        encoded = self.step_encoder(input_hidden, src_key_padding_mask=attention_mask)
        return encoded
    
    def generate_reasoning_step(
        self,
        context: torch.Tensor,
        previous_steps: List[torch.Tensor],
        step_num: int
    ) -> Tuple[torch.Tensor, float, ReasoningType]:
        """Generate a single reasoning step"""
        # Combine context with previous steps
        if previous_steps:
            prev_tensor = torch.stack(previous_steps, dim=1)
            combined = torch.cat([context.unsqueeze(1), prev_tensor], dim=1)
        else:
            combined = context.unsqueeze(1)
        
        # Add position embedding
        batch_size = context.size(0)
        position = torch.tensor([step_num]).expand(batch_size).to(context.device)
        pos_emb = self.step_position_embeddings(position)
        
        # Generate step representation
        output, _ = self.step_generator(combined)
        step_repr = output[:, -1, :]  # Last position
        
        # Add position information
        step_repr = step_repr + pos_emb.unsqueeze(1).expand(-1, step_repr.size(1), -1).mean(dim=2)
        
        # Predict step type
        step_type_logits = self.step_type_head(step_repr)
        step_type_idx = step_type_logits.argmax(dim=-1)
        step_type = list(ReasoningType)[step_type_idx.item()]
        
        # Predict validity
        validity = self.validity_head(step_repr).squeeze(-1)
        
        return step_repr, validity.item(), step_type
    
    def verify_step_consistency(
        self,
        current_step: torch.Tensor,
        previous_steps: List[torch.Tensor],
        step_type: ReasoningType
    ) -> float:
        """Verify consistency of current step with previous steps"""
        if not previous_steps:
            return 1.0
        
        # Check logical consistency
        consistency_scores = []
        
        for prev_step in previous_steps[-3:]:  # Check last 3 steps
            # Compute similarity
            similarity = F.cosine_similarity(
                current_step.unsqueeze(0),
                prev_step.unsqueeze(0),
                dim=-1
            ).item()
            
            # Steps should be related but not identical
            if 0.3 < similarity < 0.9:
                consistency_scores.append(1.0)
            elif similarity > 0.95:
                consistency_scores.append(0.3)  # Too similar, might be repetition
            else:
                consistency_scores.append(0.7)  # Somewhat consistent
        
        return np.mean(consistency_scores) if consistency_scores else 1.0

class ChainOfThoughtGenerator(nn.Module):
    """Generates explicit chain-of-thought reasoning"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Thought chain LSTM
        self.thought_chain = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=3,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Thought templates
        self.thought_templates = nn.Embedding(
            20,  # Number of thought templates
            config.hidden_size
        )
        
        # Next thought predictor
        self.next_thought_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
    
    def generate_thought_chain(
        self,
        problem_encoding: torch.Tensor,
        max_thoughts: int = 10
    ) -> List[Tuple[torch.Tensor, str]]:
        """Generate chain of thoughts for problem solving"""
        thoughts = []
        hidden = None
        
        current_input = problem_encoding.unsqueeze(1)
        
        for i in range(max_thoughts):
            # Generate next thought
            output, hidden = self.thought_chain(current_input, hidden)
            thought_repr = output[:, -1, :]
            
            # Apply template
            template_idx = min(i, 19)  # Cap at available templates
            template = self.thought_templates(
                torch.tensor([template_idx]).to(problem_encoding.device)
            )
            
            thought_repr = thought_repr + template
            
            # Generate thought text (simplified - in practice, use decoder)
            thought_type = self._classify_thought_type(thought_repr, i)
            thoughts.append((thought_repr, thought_type))
            
            # Prepare for next thought
            current_input = self.next_thought_head(thought_repr).unsqueeze(1)
            
            # Check if we should stop
            if self._should_stop_thinking(thoughts):
                break
        
        return thoughts
    
    def _classify_thought_type(self, thought_repr: torch.Tensor, step: int) -> str:
        """Classify the type of thought based on representation"""
        thought_types = [
            "Let me understand the problem",
            "Breaking this down into parts",
            "The key insight here is",
            "I need to consider",
            "Applying the relevant concept",
            "Checking my reasoning",
            "Therefore, the answer is"
        ]
        
        # Simple heuristic based on step number
        if step == 0:
            return thought_types[0]
        elif step < 3:
            return thought_types[1]
        elif step < 6:
            return thought_types[np.random.choice([2, 3, 4])]
        else:
            return thought_types[np.random.choice([5, 6])]
    
    def _should_stop_thinking(self, thoughts: List[Tuple[torch.Tensor, str]]) -> bool:
        """Determine if we should stop generating thoughts"""
        if len(thoughts) < 3:
            return False
        
        # Check if we've reached a conclusion
        last_thought_type = thoughts[-1][1]
        if "answer is" in last_thought_type.lower():
            return True
        
        # Check for repetition
        if len(thoughts) > 5:
            recent_reprs = [t[0] for t in thoughts[-3:]]
            similarities = []
            for i in range(len(recent_reprs) - 1):
                sim = F.cosine_similarity(
                    recent_reprs[i].unsqueeze(0),
                    recent_reprs[i+1].unsqueeze(0),
                    dim=-1
                ).item()
                similarities.append(sim)
            
            if np.mean(similarities) > 0.95:  # High repetition
                return True
        
        return False

class StepVerifier(nn.Module):
    """Verifies the correctness of reasoning steps"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Verification network
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 3)  # Correct, Incorrect, Uncertain
        )
        
        # Error detection
        self.error_detector = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 5)  # Types of errors
        )
    
    def verify_step(
        self,
        step: torch.Tensor,
        context: torch.Tensor,
        previous_step: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Verify a single reasoning step"""
        # Prepare input
        if previous_step is not None:
            verifier_input = torch.cat([context, previous_step, step], dim=-1)
        else:
            # Pad with zeros if no previous step
            zeros = torch.zeros_like(context)
            verifier_input = torch.cat([context, zeros, step], dim=-1)
        
        # Verify correctness
        verification_logits = self.verifier(verifier_input)
        verification_probs = F.softmax(verification_logits, dim=-1)
        
        correctness = verification_probs[:, 0].item()  # Probability of being correct
        uncertainty = verification_probs[:, 2].item()
        
        # Detect potential errors
        if previous_step is not None:
            error_input = torch.cat([previous_step, step], dim=-1)
            error_logits = self.error_detector(error_input)
            error_probs = F.softmax(error_logits, dim=-1)
            
            error_types = ["logical", "arithmetic", "factual", "consistency", "other"]
            top_error_idx = error_probs.argmax(dim=-1).item()
            top_error = error_types[top_error_idx]
            error_confidence = error_probs[:, top_error_idx].item()
        else:
            top_error = None
            error_confidence = 0.0
        
        return {
            "correctness": correctness,
            "uncertainty": uncertainty,
            "confidence": correctness * (1 - uncertainty),
            "error_type": top_error if error_confidence > 0.5 else None,
            "error_confidence": error_confidence
        }
    
    def verify_chain(
        self,
        steps: List[torch.Tensor],
        context: torch.Tensor
    ) -> Dict[str, Any]:
        """Verify an entire reasoning chain"""
        step_verifications = []
        
        for i, step in enumerate(steps):
            prev_step = steps[i-1] if i > 0 else None
            verification = self.verify_step(step, context, prev_step)
            step_verifications.append(verification)
        
        # Aggregate results
        avg_correctness = np.mean([v["correctness"] for v in step_verifications])
        avg_confidence = np.mean([v["confidence"] for v in step_verifications])
        
        # Find weakest link
        min_confidence_idx = np.argmin([v["confidence"] for v in step_verifications])
        weakest_step = {
            "index": min_confidence_idx,
            "confidence": step_verifications[min_confidence_idx]["confidence"],
            "error": step_verifications[min_confidence_idx]["error_type"]
        }
        
        # Check overall consistency
        if len(steps) > 1:
            first_last_similarity = F.cosine_similarity(
                steps[0].unsqueeze(0),
                steps[-1].unsqueeze(0),
                dim=-1
            ).item()
            
            # Should have some progression from first to last
            consistency_score = 1.0 - abs(first_last_similarity - 0.5) * 2
        else:
            consistency_score = 1.0
        
        return {
            "average_correctness": avg_correctness,
            "average_confidence": avg_confidence,
            "consistency_score": consistency_score,
            "weakest_step": weakest_step,
            "step_verifications": step_verifications,
            "overall_valid": avg_confidence > self.config.min_step_confidence
        }

class MathematicalReasoningHead(nn.Module):
    """Specialized head for mathematical reasoning"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Equation parser
        self.equation_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Operation predictor
        self.operation_head = nn.Linear(
            config.hidden_size * 2,
            20  # Common mathematical operations
        )
        
        # Numerical computation module
        self.numerical_processor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)  # Numerical result
        )
        
        # Symbolic manipulation interface
        if config.use_symbolic_math:
            self.symbolic_enabled = True
        else:
            self.symbolic_enabled = False
    
    def forward(
        self,
        problem_encoding: torch.Tensor,
        problem_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform mathematical reasoning"""
        # Encode mathematical structure
        math_encoding, _ = self.equation_encoder(problem_encoding.unsqueeze(1))
        math_repr = math_encoding.squeeze(1)
        
        # Predict required operations
        operation_logits = self.operation_head(math_repr)
        operations = self._decode_operations(operation_logits)
        
        # Extract mathematical expressions if text provided
        if problem_text and self.symbolic_enabled:
            expressions = self._extract_math_expressions(problem_text)
            symbolic_results = self._solve_symbolically(expressions, operations)
        else:
            expressions = []
            symbolic_results = {}
        
        # Numerical estimation
        numerical_estimate = self.numerical_processor(math_repr)
        
        return {
            "operations": operations,
            "expressions": expressions,
            "symbolic_results": symbolic_results,
            "numerical_estimate": numerical_estimate.item(),
            "reasoning_type": "mathematical"
        }
    
    def _decode_operations(self, logits: torch.Tensor) -> List[str]:
        """Decode mathematical operations from logits"""
        operations = [
            "addition", "subtraction", "multiplication", "division",
            "exponentiation", "root", "logarithm", "derivative",
            "integral", "limit", "summation", "product",
            "factorial", "combination", "permutation", "modulo",
            "gcd", "lcm", "trigonometry", "inverse"
        ]
        
        probs = F.softmax(logits, dim=-1)
        top_k = 5
        top_indices = probs.topk(top_k, dim=-1).indices[0]
        
        return [operations[idx] for idx in top_indices.tolist()]
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text"""
        expressions = []
        
        # Pattern for equations
        equation_patterns = [
            r'(\d+\s*[+\-*/]\s*\d+)',  # Basic arithmetic
            r'([a-zA-Z]\s*=\s*[^,;.]+)',  # Variable assignments
            r'(\d+\s*[a-zA-Z]\s*[+\-*/]\s*\d+\s*[a-zA-Z])',  # Algebraic
            r'([a-zA-Z]+\([^)]+\))',  # Functions
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        return expressions
    
    def _solve_symbolically(
        self,
        expressions: List[str],
        operations: List[str]
    ) -> Dict[str, Any]:
        """Solve expressions symbolically using SymPy"""
        if not self.symbolic_enabled:
            return {}
        
        results = {}
        
        for expr in expressions:
            try:
                # Parse expression
                parsed = sympy.sympify(expr)
                
                # Apply relevant operations
                if "derivative" in operations and hasattr(parsed, 'diff'):
                    results[f"d/dx({expr})"] = str(parsed.diff('x'))
                
                if "integral" in operations and hasattr(parsed, 'integrate'):
                    results[f"∫({expr})dx"] = str(parsed.integrate('x'))
                
                # Simplify
                simplified = sympy.simplify(parsed)
                if simplified != parsed:
                    results[f"simplify({expr})"] = str(simplified)
                
                # Solve if it's an equation
                if isinstance(parsed, sympy.Eq):
                    solutions = sympy.solve(parsed)
                    results[f"solve({expr})"] = str(solutions)
                
            except Exception as e:
                logger.debug(f"Failed to process expression {expr}: {e}")
        
        return results

class LogicalReasoningHead(nn.Module):
    """Specialized head for logical reasoning and inference"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Premise encoder
        self.premise_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Logical operator predictor
        self.logic_operator_head = nn.Linear(
            config.hidden_size,
            10  # AND, OR, NOT, IMPLIES, IFF, XOR, etc.
        )
        
        # Inference engine
        self.inference_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, 3)  # Valid, Invalid, Uncertain
        )
        
        # Logical form generator
        self.logical_form_head = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(
        self,
        premises: List[torch.Tensor],
        conclusion: torch.Tensor
    ) -> Dict[str, Any]:
        """Perform logical reasoning"""
        # Encode premises
        if premises:
            premise_tensor = torch.stack(premises, dim=1)
            encoded_premises = self.premise_encoder(premise_tensor)
            premise_repr = encoded_premises.mean(dim=1)  # Pool across premises
        else:
            premise_repr = torch.zeros_like(conclusion)
        
        # Predict logical operators
        operator_logits = self.logic_operator_head(premise_repr)
        operators = self._decode_operators(operator_logits)
        
        # Perform inference
        inference_input = torch.cat([premise_repr, conclusion], dim=-1)
        inference_logits = self.inference_network(inference_input)
        inference_result = F.softmax(inference_logits, dim=-1)
        
        # Generate logical form
        logical_form = self.logical_form_head(premise_repr)
        
        # Check logical patterns
        patterns = self._check_logical_patterns(premises, conclusion)
        
        return {
            "validity": inference_result[:, 0].item(),
            "invalidity": inference_result[:, 1].item(),
            "uncertainty": inference_result[:, 2].item(),
            "operators": operators,
            "logical_patterns": patterns,
            "logical_form": logical_form,
            "reasoning_type": "logical"
        }
    
    def _decode_operators(self, logits: torch.Tensor) -> List[str]:
        """Decode logical operators"""
        operators = [
            "AND", "OR", "NOT", "IMPLIES", "IFF",
            "XOR", "NAND", "NOR", "FORALL", "EXISTS"
        ]
        
        probs = F.softmax(logits, dim=-1)
        top_indices = probs.topk(3, dim=-1).indices[0]
        
        return [operators[idx] for idx in top_indices.tolist()]
    
    def _check_logical_patterns(
        self,
        premises: List[torch.Tensor],
        conclusion: torch.Tensor
    ) -> Dict[str, bool]:
        """Check for common logical patterns"""
        patterns = {}
        
        if len(premises) >= 2:
            # Modus Ponens: P, P->Q |- Q
            patterns["modus_ponens"] = self._check_modus_ponens(premises, conclusion)
            
            # Modus Tollens: P->Q, ~Q |- ~P
            patterns["modus_tollens"] = self._check_modus_tollens(premises, conclusion)
            
            # Hypothetical Syllogism: P->Q, Q->R |- P->R
            patterns["hypothetical_syllogism"] = self._check_syllogism(premises, conclusion)
        
        # Contradiction check
        patterns["contradiction"] = self._check_contradiction(premises)
        
        return patterns
    
    def _check_modus_ponens(
        self,
        premises: List[torch.Tensor],
        conclusion: torch.Tensor
    ) -> bool:
        """Check if reasoning follows modus ponens pattern"""
        # Simplified check based on representations
        if len(premises) < 2:
            return False
        
        # Check if first premise appears in conclusion
        similarity = F.cosine_similarity(
            premises[0].unsqueeze(0),
            conclusion.unsqueeze(0),
            dim=-1
        ).item()
        
        return similarity > 0.7
    
    def _check_modus_tollens(
        self,
        premises: List[torch.Tensor],
        conclusion: torch.Tensor
    ) -> bool:
        """Check if reasoning follows modus tollens pattern"""
        # Simplified check
        if len(premises) < 2:
            return False
        
        # Check for negation pattern
        # In practice, this would be more sophisticated
        return False
    
    def _check_syllogism(
        self,
        premises: List[torch.Tensor],
        conclusion: torch.Tensor
    ) -> bool:
        """Check if reasoning follows syllogistic pattern"""
        if len(premises) < 2:
            return False
        
        # Check transitivity
        p1_p2_sim = F.cosine_similarity(
            premises[0].unsqueeze(0),
            premises[1].unsqueeze(0),
            dim=-1
        ).item()
        
        p1_c_sim = F.cosine_similarity(
            premises[0].unsqueeze(0),
            conclusion.unsqueeze(0),
            dim=-1
        ).item()
        
        return p1_p2_sim > 0.5 and p1_c_sim > 0.5
    
    def _check_contradiction(self, premises: List[torch.Tensor]) -> bool:
        """Check if premises contain contradictions"""
        for i in range(len(premises)):
            for j in range(i + 1, len(premises)):
                similarity = F.cosine_similarity(
                    premises[i].unsqueeze(0),
                    premises[j].unsqueeze(0),
                    dim=-1
                ).item()
                
                # Very negative similarity might indicate contradiction
                if similarity < -0.7:
                    return True
        
        return False

class ScientificReasoningHead(nn.Module):
    """Specialized head for scientific reasoning"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Hypothesis generator
        self.hypothesis_generator = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Evidence evaluator
        self.evidence_evaluator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 4)  # Support levels
        )
        
        # Causal relationship detector
        self.causal_detector = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3)  # Causal, Correlational, None
        )
        
        # Domain classifier
        self.domain_classifier = nn.Linear(
            config.hidden_size,
            10  # Physics, Chemistry, Biology, etc.
        )
    
    def forward(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Perform scientific reasoning"""
        # Generate hypothesis
        hypothesis, _ = self.hypothesis_generator(observation.unsqueeze(1))
        hypothesis_repr = hypothesis.squeeze(1)
        
        # Classify domain
        domain_logits = self.domain_classifier(observation)
        domain = self._decode_domain(domain_logits)
        
        # Evaluate evidence strength
        if context is not None:
            evidence_input = torch.cat([observation, context], dim=-1)
            evidence_logits = self.evidence_evaluator(evidence_input)
            evidence_strength = F.softmax(evidence_logits, dim=-1)
        else:
            evidence_strength = None
        
        # Detect causal relationships
        if context is not None:
            causal_input = torch.cat([observation, context], dim=-1)
            causal_logits = self.causal_detector(causal_input)
            causal_type = self._decode_causal_type(causal_logits)
        else:
            causal_type = "none"
        
        # Apply domain-specific reasoning
        domain_results = self._apply_domain_reasoning(
            observation,
            hypothesis_repr,
            domain
        )
        
        return {
            "hypothesis": hypothesis_repr,
            "domain": domain,
            "evidence_strength": evidence_strength,
            "causal_relationship": causal_type,
            "domain_specific": domain_results,
            "reasoning_type": "scientific"
        }
    
    def _decode_domain(self, logits: torch.Tensor) -> str:
        """Decode scientific domain"""
        domains = [
            "physics", "chemistry", "biology", "astronomy",
            "earth_science", "computer_science", "mathematics",
            "engineering", "medicine", "psychology"
        ]
        
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1).item()
        
        return domains[top_idx]
    
    def _decode_causal_type(self, logits: torch.Tensor) -> str:
        """Decode causal relationship type"""
        types = ["causal", "correlational", "none"]
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1).item()
        
        return types[top_idx]
    
    def _apply_domain_reasoning(
        self,
        observation: torch.Tensor,
        hypothesis: torch.Tensor,
        domain: str
    ) -> Dict[str, Any]:
        """Apply domain-specific reasoning rules"""
        results = {}
        
        if domain == "physics":
            results["conservation_laws"] = self._check_conservation_laws(observation)
            results["dimensional_analysis"] = True  # Placeholder
        elif domain == "chemistry":
            results["reaction_balanced"] = True  # Placeholder
            results["thermodynamically_favorable"] = True  # Placeholder
        elif domain == "biology":
            results["evolutionary_consistent"] = True  # Placeholder
            results["mechanistically_plausible"] = True  # Placeholder
        
        return results
    
    def _check_conservation_laws(self, observation: torch.Tensor) -> bool:
        """Check if observation respects conservation laws"""
        # Simplified check - in practice would be more sophisticated
        return True

class CodeReasoningHead(nn.Module):
    """Specialized head for code understanding and generation"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Code structure encoder
        self.code_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Algorithm pattern detector
        self.pattern_detector = nn.Linear(
            config.hidden_size,
            20  # Common algorithmic patterns
        )
        
        # Complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 5)  # O(1), O(log n), O(n), O(n log n), O(n²)
        )
        
        # Bug detector
        self.bug_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 10)  # Common bug types
        )
    
    def forward(
        self,
        code_repr: torch.Tensor,
        problem_context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Perform code reasoning"""
        # Encode code structure
        encoded = self.code_encoder(code_repr.unsqueeze(1))
        code_encoding = encoded.squeeze(1)
        
        # Detect algorithmic patterns
        pattern_logits = self.pattern_detector(code_encoding)
        patterns = self._decode_patterns(pattern_logits)
        
        # Analyze complexity
        complexity_logits = self.complexity_analyzer(code_encoding)
        complexity = self._decode_complexity(complexity_logits)
        
        # Detect potential bugs
        bug_logits = self.bug_detector(code_encoding)
        bugs = self._decode_bugs(bug_logits)
        
        # Generate algorithm steps
        algorithm_steps = self._generate_algorithm_steps(patterns, complexity)
        
        return {
            "patterns": patterns,
            "complexity": complexity,
            "potential_bugs": bugs,
            "algorithm_steps": algorithm_steps,
            "reasoning_type": "code"
        }
    
    def _decode_patterns(self, logits: torch.Tensor) -> List[str]:
        """Decode algorithmic patterns"""
        patterns = [
            "iteration", "recursion", "divide_conquer", "dynamic_programming",
            "greedy", "backtracking", "graph_traversal", "tree_traversal",
            "sorting", "searching", "hashing", "two_pointers",
            "sliding_window", "union_find", "bit_manipulation", "math",
            "string_manipulation", "stack", "queue", "heap"
        ]
        
        probs = F.softmax(logits, dim=-1)
        threshold = 0.3
        detected = []
        
        for i, p in enumerate(probs[0].tolist()):
            if p > threshold:
                detected.append(patterns[i])
        
        return detected
    
    def _decode_complexity(self, logits: torch.Tensor) -> str:
        """Decode time complexity"""
        complexities = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)"]
        
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1).item()
        
        return complexities[top_idx]
    
    def _decode_bugs(self, logits: torch.Tensor) -> List[str]:
        """Decode potential bugs"""
        bug_types = [
            "off_by_one", "null_pointer", "infinite_loop", "array_bounds",
            "type_error", "logic_error", "race_condition", "memory_leak",
            "unhandled_exception", "incorrect_initialization"
        ]
        
        probs = F.softmax(logits, dim=-1)
        threshold = 0.4
        detected = []
        
        for i, p in enumerate(probs[0].tolist()):
            if p > threshold:
                detected.append(bug_types[i])
        
        return detected
    
    def _generate_algorithm_steps(
        self,
        patterns: List[str],
        complexity: str
    ) -> List[str]:
        """Generate high-level algorithm steps based on patterns"""
        steps = []
        
        if "iteration" in patterns:
            steps.append("Iterate through the input elements")
        
        if "recursion" in patterns:
            steps.append("Define base case for recursion")
            steps.append("Make recursive calls with smaller subproblems")
        
        if "divide_conquer" in patterns:
            steps.append("Divide problem into smaller subproblems")
            steps.append("Solve subproblems recursively")
            steps.append("Combine results from subproblems")
        
        if "dynamic_programming" in patterns:
            steps.append("Identify overlapping subproblems")
            steps.append("Define state and recurrence relation")
            steps.append("Build solution bottom-up or use memoization")
        
        if "graph_traversal" in patterns:
            steps.append("Initialize visited set/array")
            steps.append("Traverse graph using BFS/DFS")
            steps.append("Process each node/edge")
        
        if not steps:
            steps.append("Process input according to problem requirements")
        
        steps.append(f"Time complexity: {complexity}")
        
        return steps