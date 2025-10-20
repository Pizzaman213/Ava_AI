#!/usr/bin/env python3
"""
Download and process conversational datasets with greetings and natural dialogue.
Focus on datasets with "hello", greetings, and back-and-forth conversations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system("pip install -q datasets tqdm")
    from datasets import load_dataset
    from tqdm import tqdm


# Output directory
OUTPUT_DIR = Path("/project/code/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_conversational_dataset(name: str, config: Optional[str] = None, max_samples: int = 50000):
    """Download and process conversational datasets with greetings."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        if config:
            dataset = load_dataset(name, config, split="train", streaming=True)
        else:
            dataset = load_dataset(name, split="train", streaming=True)

        # Output file
        safe_name = name.replace("/", "_").replace("-", "_")
        output_file = OUTPUT_DIR / f"{safe_name}_conversational.jsonl"

        count = 0
        saved_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc=f"Processing {name}", total=max_samples):
                if count >= max_samples:
                    break

                # Extract conversational text based on dataset structure
                text = extract_conversation(item, name)

                if text and len(text.strip()) > 20:
                    # Prefer conversations with greetings
                    has_greeting = has_greeting_words(text)

                    # Save all conversations, but prioritize those with greetings
                    if has_greeting or saved_count < max_samples * 0.3:  # Keep 30% non-greeting too
                        record = {
                            "text": text.strip(),
                            "source": name,
                            "has_greeting": has_greeting
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        saved_count += 1

                count += 1

        print(f"✓ Saved {saved_count} examples to {output_file}")
        print(f"  Total processed: {count}, Kept: {saved_count} ({saved_count/count*100:.1f}%)")

        # Show file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

        return saved_count

    except Exception as e:
        print(f"✗ Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def extract_conversation(item: Dict, dataset_name: str) -> str:
    """Extract conversational text from dataset item."""

    # Handle different conversation formats
    if 'messages' in item:
        # OpenAI-style messages format
        messages = item['messages']
        if isinstance(messages, list):
            conversation_parts = []
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                conversation_parts.append(f"{role.capitalize()}: {content}")
            return "\n".join(conversation_parts)

    elif 'conversations' in item:
        # Conversation list format
        convs = item['conversations']
        if isinstance(convs, list):
            conversation_parts = []
            for conv in convs:
                if isinstance(conv, dict):
                    role = conv.get('from', conv.get('role', 'unknown'))
                    content = conv.get('value', conv.get('content', ''))
                    conversation_parts.append(f"{role.capitalize()}: {content}")
            return "\n".join(conversation_parts)

    elif 'prompt' in item and 'response' in item:
        # Prompt-response format
        prompt = item['prompt']
        response = item['response']
        return f"User: {prompt}\nAssistant: {response}"

    elif 'instruction' in item and 'response' in item:
        # Instruction-response format
        instruction = item['instruction']
        response = item['response']
        context = item.get('context', '')
        if context:
            return f"User: {instruction}\nContext: {context}\nAssistant: {response}"
        return f"User: {instruction}\nAssistant: {response}"

    elif 'chosen' in item and 'rejected' in item:
        # RLHF format - use chosen response
        chosen = item['chosen']
        if isinstance(chosen, list):
            conversation_parts = []
            for msg in chosen:
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    conversation_parts.append(f"{role.capitalize()}: {content}")
            return "\n".join(conversation_parts)

    elif 'text' in item:
        # Plain text format
        return item['text']

    # Try to find any conversation-like structure
    for key in ['dialogue', 'conversation', 'chat', 'turns']:
        if key in item:
            return str(item[key])

    # Fallback: combine all text fields
    text_parts = []
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 10 and key not in ['id', 'source', 'dataset']:
            text_parts.append(f"{key.capitalize()}: {value}")

    return "\n".join(text_parts) if text_parts else ""


def has_greeting_words(text: str) -> bool:
    """Check if text contains greeting words."""
    text_lower = text.lower()

    # Common greetings
    greetings = [
        'hello', 'hi ', 'hi,', 'hi!', 'hey', 'greetings', 'good morning',
        'good afternoon', 'good evening', 'howdy', "what's up", 'whats up',
        'how are you', 'how do you do', 'nice to meet', 'pleased to meet',
        'welcome', 'salutations', 'hiya', 'hola', 'bonjour', 'ciao',
        'aloha', 'namaste', 'sup ', 'yo ', 'heya'
    ]

    return any(greeting in text_lower for greeting in greetings)


def download_conversational_datasets():
    """Download popular conversational datasets."""

    datasets_to_download = [
        {
            "name": "HuggingFaceH4/ultrachat_200k",
            "max_samples": 200000,
            "description": "High-quality multi-turn conversations"
        },
        {
            "name": "Anthropic/hh-rlhf",
            "max_samples": 160000,
            "description": "Human feedback conversations with greetings"
        },
        {
            "name": "OpenAssistant/oasst1",
            "max_samples": 100000,
            "description": "Open Assistant conversational threads"
        },
        {
            "name": "HuggingFaceH4/no_robots",
            "max_samples": 10000,
            "description": "Human-written dialogues"
        },
        {
            "name": "daily_dialog",
            "max_samples": 13000,
            "description": "Daily conversations covering greetings and small talk"
        },
        {
            "name": "empathetic_dialogues",
            "max_samples": 25000,
            "description": "Empathetic conversations with emotional context"
        },
        {
            "name": "AlekseyKorshuk/persona-chat",
            "max_samples": 10000,
            "description": "Persona-based chit-chat conversations"
        },
        {
            "name": "blended_skill_talk",
            "max_samples": 5000,
            "description": "Multi-skill conversations including greetings"
        },
        {
            "name": "conv_ai_2",
            "max_samples": 10000,
            "description": "Conversational AI dialogue dataset"
        },
        {
            "name": "google/Synthetic-Persona-Chat",
            "max_samples": 20000,
            "description": "Synthetic persona-based conversations"
        },
    ]

    print("\n" + "="*60)
    print("DOWNLOADING CONVERSATIONAL DATASETS WITH GREETINGS")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total datasets to process: {len(datasets_to_download)}")
    print("="*60)

    total_examples = 0
    successful = 0

    for dataset_info in datasets_to_download:
        name = dataset_info["name"]
        max_samples = dataset_info.get("max_samples", 10000)
        config = dataset_info.get("config", None)

        print(f"\n📦 {dataset_info['description']}")

        count = process_conversational_dataset(name, config, max_samples)

        if count > 0:
            total_examples += count
            successful += 1

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successful: {successful}/{len(datasets_to_download)} datasets")
    print(f"✓ Total examples collected: {total_examples:,}")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60)

    # List all conversational files
    print("\nCreated files:")
    total_size = 0
    for file in sorted(OUTPUT_DIR.glob("*_conversational.jsonl")):
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {file.name} ({size_mb:.1f} MB)")

    print(f"\nTotal size: {total_size:.1f} MB")

    return total_examples


def create_custom_greeting_conversations():
    """Create custom conversational examples with greetings."""

    output_file = OUTPUT_DIR / "custom_greeting_conversations.jsonl"

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
    print(f"\n✓ Created custom greeting conversations: {output_file}")
    print(f"  Total conversations: {len(conversations)}")
    print(f"  File size: {size_mb:.2f} MB")

    return len(conversations)


if __name__ == "__main__":
    print("Starting conversational dataset collection...")
    print("Focus: Greetings, hellos, and natural dialogue\n")

    # Create custom greeting examples first
    custom_count = create_custom_greeting_conversations()

    # Download and process conversational datasets
    downloaded_count = download_conversational_datasets()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✓ Custom greeting conversations: {custom_count}")
    print(f"✓ Downloaded conversational examples: {downloaded_count:,}")
    print(f"✓ Total new conversational examples: {custom_count + downloaded_count:,}")
    print("="*60)
    print("\n📁 All files saved to: /project/code/data/processed/")
    print("\nThese datasets will be automatically loaded when you run training")
    print("with data_dir: /project/code/data/processed")
    print("\nThe system will mix these conversational datasets with your")
    print("existing datasets for balanced training.")
    print("="*60)
