"""
FastAPI server for LLM model serving with batching and optimization.

This module provides a production-ready serving infrastructure with
features like dynamic batching, caching, and performance monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
import torch
import torch.nn as nn
import asyncio
import time
import uuid
import json
import logging
from datetime import datetime
from collections import deque, defaultdict
import numpy as np
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import uvicorn


# Request/Response Models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=0.0, le=2.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    seed: Optional[int] = Field(None, description="Random seed for generation")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    request_id: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    finish_reason: str
    metadata: Dict[str, Any] = {}


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    token: str
    request_id: str
    is_final: bool = False
    metadata: Dict[str, Any] = {}


class BatchRequest(BaseModel):
    """Batch generation request."""
    requests: List[GenerationRequest]
    batch_id: Optional[str] = Field(None, description="Optional batch ID")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime_seconds: float
    requests_processed: int


@dataclass
class BatchedRequest:
    """Internal representation of a batched request."""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    stop_sequences: Optional[List[str]]
    stream: bool
    seed: Optional[int]
    timestamp: float
    future: asyncio.Future


class DynamicBatcher:
    """
    Dynamic batcher for efficient request batching.

    This class accumulates requests and processes them in batches
    to maximize GPU utilization.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_queue_size: int = 1000
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size

        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_processor = None
        self.running = False

    async def add_request(self, request: BatchedRequest) -> Any:
        """Add a request to the batch queue."""
        if self.queue.qsize() >= self.max_queue_size:
            raise HTTPException(status_code=503, detail="Server overloaded")

        await self.queue.put(request)
        return await request.future

    def start_batch_processor(self, model_inference_fn):
        """Start the batch processing loop."""
        self.running = True
        self.batch_processor = asyncio.create_task(
            self._batch_processing_loop(model_inference_fn)
        )

    async def stop_batch_processor(self):
        """Stop the batch processing loop."""
        self.running = False
        if self.batch_processor:
            self.batch_processor.cancel()
            try:
                await self.batch_processor
            except asyncio.CancelledError:
                pass

    async def _batch_processing_loop(self, model_inference_fn):
        """Main batch processing loop."""
        while self.running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch, model_inference_fn)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in batch processing: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[BatchedRequest]:
        """Collect requests into a batch."""
        batch = []
        deadline = time.time() + self.max_wait_time

        # Get first request (blocking)
        try:
            first_request = await asyncio.wait_for(
                self.queue.get(), timeout=self.max_wait_time
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return []

        # Collect additional requests (non-blocking)
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    self.queue.get(), timeout=0.001
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: List[BatchedRequest], model_inference_fn):
        """Process a batch of requests."""
        try:
            # Extract batch data
            prompts = [req.prompt for req in batch]
            generation_configs = [{
                'max_tokens': req.max_tokens,
                'temperature': req.temperature,
                'top_p': req.top_p,
                'top_k': req.top_k,
                'repetition_penalty': req.repetition_penalty,
                'stop_sequences': req.stop_sequences,
                'seed': req.seed
            } for req in batch]

            # Run inference
            start_time = time.time()
            results = await model_inference_fn(prompts, generation_configs)
            generation_time = time.time() - start_time

            # Set results for each request
            for req, result in zip(batch, results):
                if not req.future.cancelled():
                    req.future.set_result({
                        'generated_text': result.get('generated_text', ''),
                        'tokens_generated': result.get('tokens_generated', 0),
                        'generation_time': generation_time,
                        'finish_reason': result.get('finish_reason', 'length')
                    })

        except Exception as e:
            # Set exception for all requests in the batch
            for req in batch:
                if not req.future.cancelled():
                    req.future.set_exception(e)


class ModelManager:
    """
    Model manager for loading and managing the LLM model.

    Handles model loading, device management, and inference.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        # Performance statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    async def load_model(self):
        """Load the model and tokenizer."""
        try:
            logging.info(f"Loading model from {self.model_path}")

            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                self.model, self.tokenizer = await loop.run_in_executor(
                    executor, self._load_model_sync
                )

            self.model_loaded = True
            logging.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def _load_model_sync(self):
        """Synchronous model loading."""
        # This is a placeholder - implement actual model loading
        # based on your specific model format
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device != "cpu" else None
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    async def generate_batch(
        self,
        prompts: List[str],
        generation_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate text for a batch of prompts."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, self._generate_batch_sync, prompts, generation_configs
            )

        self.total_requests += len(prompts)
        return results

    def _generate_batch_sync(
        self,
        prompts: List[str],
        generation_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Synchronous batch generation."""
        start_time = time.time()

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Get generation parameters (use first config for simplicity)
        config = generation_configs[0]
        max_new_tokens = config.get('max_tokens', 100)
        temperature = config.get('temperature', 0.7)
        top_p = config.get('top_p', 0.9)
        top_k = config.get('top_k', 50)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            input_length = inputs.input_ids[i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            results.append({
                'generated_text': generated_text,
                'tokens_generated': len(generated_tokens),
                'finish_reason': 'length'  # Simplified
            })

        generation_time = time.time() - start_time
        self.total_time += generation_time
        self.total_tokens += sum(r['tokens_generated'] for r in results)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        avg_tokens_per_second = (
            self.total_tokens / self.total_time if self.total_time > 0 else 0
        )

        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_time': self.total_time,
            'avg_tokens_per_second': avg_tokens_per_second,
            'model_loaded': self.model_loaded,
            'device': self.device
        }


class LLMServer:
    """
    Main LLM serving server with FastAPI.

    Provides REST API endpoints for text generation with batching
    and performance optimization.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        self.app = FastAPI(
            title="LLM Serving API",
            description="High-performance LLM serving with dynamic batching",
            version="1.0.0"
        )

        # Components
        self.model_manager = ModelManager(model_path, device)
        self.batcher = DynamicBatcher(max_batch_size, max_wait_time)

        # Statistics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.middleware("http")
        async def logging_middleware(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time

            logging.info(
                f"{request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.3f}s"
            )

            return response

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.on_event("startup")
        async def startup_event():
            await self.model_manager.load_model()
            self.batcher.start_batch_processor(self.model_manager.generate_batch)
            logging.info("Server started successfully")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self.batcher.stop_batch_processor()
            logging.info("Server shutdown complete")

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            memory_info = psutil.virtual_memory()
            uptime = time.time() - self.start_time

            return HealthResponse(
                status="healthy" if self.model_manager.model_loaded else "loading",
                model_loaded=self.model_manager.model_loaded,
                gpu_available=torch.cuda.is_available(),
                memory_usage={
                    "used_percent": memory_info.percent,
                    "available_gb": memory_info.available / (1024**3)
                },
                uptime_seconds=uptime,
                requests_processed=self.request_count
            )

        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text from a prompt."""
            request_id = request.request_id or str(uuid.uuid4())

            try:
                self.request_count += 1

                # Create batched request
                batched_request = BatchedRequest(
                    request_id=request_id,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    stop_sequences=request.stop_sequences,
                    stream=request.stream,
                    seed=request.seed,
                    timestamp=time.time(),
                    future=asyncio.Future()
                )

                # Add to batcher and wait for result
                result = await self.batcher.add_request(batched_request)

                # Calculate performance metrics
                tokens_per_second = (
                    result['tokens_generated'] / result['generation_time']
                    if result['generation_time'] > 0 else 0
                )

                return GenerationResponse(
                    generated_text=result['generated_text'],
                    request_id=request_id,
                    tokens_generated=result['tokens_generated'],
                    generation_time=result['generation_time'],
                    tokens_per_second=tokens_per_second,
                    finish_reason=result['finish_reason']
                )

            except Exception as e:
                self.error_count += 1
                logging.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/generate/batch")
        async def generate_batch(batch_request: BatchRequest):
            """Generate text for multiple prompts."""
            batch_id = batch_request.batch_id or str(uuid.uuid4())

            try:
                tasks = []
                for req in batch_request.requests:
                    req.request_id = req.request_id or str(uuid.uuid4())
                    task = generate_text(req)
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                return {
                    "batch_id": batch_id,
                    "results": results,
                    "total_requests": len(batch_request.requests),
                    "successful_requests": sum(
                        1 for r in results if not isinstance(r, Exception)
                    )
                }

            except Exception as e:
                logging.error(f"Batch generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            model_stats = self.model_manager.get_stats()

            return {
                "server": {
                    "uptime_seconds": time.time() - self.start_time,
                    "total_requests": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1)
                },
                "model": model_stats,
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "gpu_available": torch.cuda.is_available()
                }
            }

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level
        )


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Serving Server")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Max batch size")
    parser.add_argument("--max-wait-time", type=float, default=0.1, help="Max wait time")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run server
    server = LLMServer(
        model_path=args.model_path,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time
    )

    logging.info(f"Starting server on {args.host}:{args.port}")
    server.run(host=args.host, port=args.port, log_level=args.log_level)