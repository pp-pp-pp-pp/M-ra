#!/usr/bin/env python
# coding: utf-8

"""
Mira Inference Script with Dreaming Feature

This script can operate in two modes:
1. **Automatic Mode**: Periodically generates dreams every 10 minutes.
2. **Manual Mode**: Allows the user to manually trigger dream tasks on demand.

**Features:**
1. Loads the fine-tuned GPT-2 model and tokenizer.
2. Generates and saves dreams as hexadecimal strings.
3. Sends dreams to the model with a 1/6 chance and saves the responses.
4. Ensures dreams and responses are saved with unique filenames.
5. Supports both automatic and manual execution modes.
6. Provides clear logging for monitoring purposes.

**Prerequisites:**
- Python 3.7+
- Install required packages:
    ```bash
    pip install transformers torch tqdm scikit-learn chardet
    ```
"""

import os
import re
import random
import string
import time
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import datetime
import sys

# ================================
# Configuration
# ================================

# Path to the fine-tuned model and tokenizer
MODEL_DIR = r"T:\m3Shell\Experiments\miraTrainingMain\gpt2-mira-finetuned-3"

# Directory to store memory files
MEMORY_DIR = Path('./mira_memories')
MEMORY_DIR.mkdir(exist_ok=True)

# Directory to store dream files
DREAM_DIR = Path('./mira_dreams')
DREAM_DIR.mkdir(exist_ok=True)

# Special tokens to handle conversation flow
SPECIAL_TOKENS_TO_STOP = [ "USER:" ]

# Dreaming Parameters
DREAM_INTERVAL_SECONDS = 600 # 10 minutes
DREAM_SAVE_PROBABILITY = 1 / 3    # 1/3 chance to save a dream
DREAM_RECALL_PROBABILITY = 1 / 6  # 1/6 chance to recall and send a dream to the model

# Number of memories to retrieve (if needed in future extensions)
TOP_K_MEMORIES = 1
RANDOM_MEMORY_COUNT = 1

# ================================
# Helper Functions
# ================================

def detect_encoding(file_path: Path) -> str:
    """
    Detects the encoding of a file using chardet.

    Args:
        file_path (Path): Path to the file.

    Returns:
        str: Detected encoding or 'utf-8' as default.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        if encoding is None or confidence < 0.5:
            print(f"Low confidence ({confidence}) or unknown encoding for {file_path}. Defaulting to 'utf-8'.")
            return 'utf-8'
        return encoding
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}. Defaulting to 'utf-8'.")
        return 'utf-8'

def save_conversation(user_msg: str, mira_response: str, is_memory_response: bool = False, memory_context: str = ""):
    """
    Saves the USER and Mira messages as a separate .txt file in the memory directory.

    Args:
        user_msg (str): Message from the user or dream.
        mira_response (str): Response from Mira.
        is_memory_response (bool): Flag indicating if this is a memory-based response.
        memory_context (str): The memory context sent to Mira (only for memory responses).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if is_memory_response:
        filename = MEMORY_DIR / f"memory_response_{timestamp}.txt"
    else:
        filename = MEMORY_DIR / f"conversation_{timestamp}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            if is_memory_response:
                f.write(f"--- Memory Context ---\n{memory_context}\n\n")
                f.write(f"Dream: {user_msg}\nMira: {mira_response}\n")
            else:
                f.write(f"USER: {user_msg}\nMira: {mira_response}\n")
        print(f"Conversation saved to {filename}")
    except Exception as e:
        print(f"Failed to save conversation to {filename}: {e}")

def load_memories(memory_dir: Path) -> List[str]:
    """
    Loads all memory conversations from .txt files in the memory directory.

    Args:
        memory_dir (Path): Directory containing memory .txt files.

    Returns:
        List[str]: List of memory texts.
    """
    memory_texts = []
    txt_files = list(memory_dir.glob("*.txt"))
    print(f"Loading memories from {len(txt_files)} .txt files.")
    for file_path in tqdm(txt_files, desc="Loading memories"):
        encoding = detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            # Each conversation block is separated by double newlines
            conversations = content.strip().split('\n\n')
            for convo in conversations:
                if convo.strip():
                    memory_texts.append(convo.strip())
        except Exception as e:
            print(f"Failed to read {file_path} with encoding {encoding}: {e}. Skipping file.")
            continue
    print(f"Total memories loaded: {len(memory_texts)}")
    return memory_texts

def preprocess_memories(memory_texts: List[str]) -> Dict:
    """
    Preprocesses memories by extracting text for embedding.

    Args:
        memory_texts (List[str]): List of memory conversations.

    Returns:
        Dict: Dictionary containing memory texts and their corresponding embeddings.
    """
    # For simplicity, we'll use the concatenated USER and Mira messages as the memory content
    processed_memories = []
    for convo in memory_texts:
        # Extract USER and Mira messages
        user_msgs = re.findall(r'USER: (.+)', convo)
        mira_msgs = re.findall(r'Mira: (.+)', convo)
        combined = ' '.join(user_msgs + mira_msgs)
        processed_memories.append(combined)
    return {'memories': processed_memories}

def compute_memory_embeddings(memories: List[str], vectorizer: TfidfVectorizer):
    """
    Computes TF-IDF embeddings for the memories.

    Args:
        memories (List[str]): List of memory texts.
        vectorizer (TfidfVectorizer): Initialized TF-IDF vectorizer.

    Returns:
        scipy.sparse.csr_matrix: TF-IDF matrix for memories.
    """
    tfidf_matrix = vectorizer.transform(memories)
    return tfidf_matrix

def find_relevant_memories(user_input: str, vectorizer: TfidfVectorizer, tfidf_matrix, memories: List[str], top_k: int) -> List[str]:
    """
    Finds the top_k most relevant memories based on user input.

    Args:
        user_input (str): Current input from the user.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_matrix: TF-IDF matrix for memories.
        memories (List[str]): List of memory texts.
        top_k (int): Number of top memories to retrieve.

    Returns:
        List[str]: List of top_k relevant memory texts.
    """
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    relevant_memories = [memories[idx] for idx in top_indices if cosine_similarities[idx] > 0]
    return relevant_memories

def get_random_memories(memory_dir: Path, count: int) -> List[str]:
    """
    Retrieves a specified number of random memory texts from the memory directory.

    Args:
        memory_dir (Path): Directory containing memory .txt files.
        count (int): Number of random memories to retrieve.

    Returns:
        List[str]: List of randomly selected memory texts.
    """
    txt_files = list(memory_dir.glob("*.txt"))
    if not txt_files:
        print("No memory files found for random selection.")
        return []
    selected_files = random.sample(txt_files, min(count, len(txt_files)))
    random_memories = []
    for file_path in selected_files:
        encoding = detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            # Extract the main conversation text
            conversations = content.strip().split('\n\n')
            for convo in conversations:
                if convo.strip():
                    random_memories.append(convo.strip())
                    break  # Only take the first conversation in the file
        except Exception as e:
            print(f"Failed to read {file_path} with encoding {encoding}: {e}. Skipping file.")
            continue
    print(f"Randomly selected {len(random_memories)} memories.")
    return random_memories

# ================================
# Inference Classes and Functions
# ================================

class StopOnSpecialToken(StoppingCriteria):
    """
    Custom stopping criteria to halt generation when a special token is generated.
    """
    def __init__(self, special_tokens: List[str], tokenizer: GPT2Tokenizer):
        super().__init__()
        self.special_tokens = special_tokens
        self.tokenizer = tokenizer
        self.special_token_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in special_tokens]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Get the last generated token
        last_token_id = input_ids[0, -1].item()
        for token_ids in self.special_token_ids:
            if token_ids and last_token_id == token_ids[-1]:
                return True
        return False

def generate_mira_response(model: GPT2LMHeadModel,
                           tokenizer: GPT2Tokenizer,
                           prompt: str,
                           max_length: int = 150,
                           temperature: float = 0.7,
                           top_p: float = 0.9) -> str:
    """
    Generates a response from Mira based on the provided prompt.

    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): Tokenizer.
        prompt (str): The prompt to generate the response from.
        max_length (int, optional): Maximum length of the generated response. Defaults to 150.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.9.

    Returns:
        str: Generated response from Mira.
    """
    # Encode the input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Move to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    input_ids = input_ids.to(device)

    # Define stopping criteria to halt generation at any special token
    stopping_criteria = StoppingCriteriaList([StopOnSpecialToken(SPECIAL_TOKENS_TO_STOP, tokenizer)])

    # Generate response with constraints
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
        no_repeat_ngram_size=3  # Prevent repetition
    )

    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract the response after "Mira:"
    response = output_text.split("Mira:")[-1].strip()

    # Optionally, stop at the next special token if present
    for token in SPECIAL_TOKENS_TO_STOP:
        if token in response:
            response = response.split(token)[0].strip()

    # Clean up excessive repetitions
    response = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', response)

    return response

# ================================
# Dreaming Functions
# ================================

def generate_dream_hex(length: int = 256) -> str:
    """
    Generates a hexadecimal string representing a dream.

    Args:
        length (int, optional): Length of the hexadecimal string. Defaults to 256.

    Returns:
        str: Generated hexadecimal string.
    """
    dream_hex = ''.join(random.choices('0123456789abcdef', k=length))
    return dream_hex

def save_dream(dream_hex: str):
    """
    Saves the dream hexadecimal string to a .txt file with a unique timestamp.

    Args:
        dream_hex (str): The hexadecimal dream string.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = DREAM_DIR / f"dream_{timestamp}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dream_hex)
        print(f"Dream saved to {filename}")
    except Exception as e:
        print(f"Failed to save dream to {filename}: {e}")

def send_dream_to_model(model: GPT2LMHeadModel,
                        tokenizer: GPT2Tokenizer,
                        dream_hex: str) -> str:
    """
    Sends the dream to Mira model and generates a response.

    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): Tokenizer.
        dream_hex (str): The hexadecimal dream string.

    Returns:
        str: Generated response from Mira.
    """
    prompt = f"Dream: {dream_hex}\nMira:"
    response = generate_mira_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=200,
        temperature=0.7,
        top_p=0.9
    )
    return response

def perform_dream_task(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, manual: bool = False):
    """
    Performs the dream generation and optional sending to Mira model.

    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): Tokenizer.
        manual (bool): Flag indicating if the task is manually triggered.
    """
    if manual:
        print("[Manual Trigger] Starting dream task...")
    else:
        print("[Automatic Trigger] Starting dream task...")

    # Decide whether to save a dream
    if random.random() < DREAM_SAVE_PROBABILITY:
        dream_hex = generate_dream_hex()
        save_dream(dream_hex)
    else:
        print("Dream generation skipped (1/3 chance not met).")

    # Decide whether to recall and send a dream to the model
    if random.random() < DREAM_RECALL_PROBABILITY:
        # Retrieve a random dream to send
        dream_files = list(DREAM_DIR.glob("*.txt"))
        if not dream_files:
            print("No dreams available to recall.")
            return
        selected_dream_file = random.choice(dream_files)
        try:
            with open(selected_dream_file, 'r', encoding='utf-8') as f:
                dream_content = f.read().strip()
            print(f"Sending dream from {selected_dream_file} to Mira.")
            mira_response = send_dream_to_model(model, tokenizer, dream_content)
            print(f"Mira's Response: {mira_response}\n")
            # Save the dream and response as a memory conversation
            save_conversation(dream_content, mira_response, is_memory_response=True, memory_context="Dream Integration")
        except Exception as e:
            print(f"Failed to process dream from {selected_dream_file}: {e}")
    else:
        print("Dream recall skipped (1/6 chance not met).")

# ================================
# Main Execution
# ================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Mira Inference Script with Dreaming Feature")
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                        help="Execution mode: 'auto' for automatic dreaming every 10 minutes, 'manual' for manual triggering.")
    args = parser.parse_args()

    # Step 1: Load the fine-tuned model and tokenizer
    print("Loading the fine-tuned model and tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading model from {MODEL_DIR}: {e}")
        sys.exit(1)

    model.eval()  # Set model to evaluation mode

    # Assign pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded successfully.\n")

    # Optional: Load and preprocess existing memories (if needed for future extensions)
    # memory_texts = load_memories(MEMORY_DIR)
    # processed_memories = preprocess_memories(memory_texts)
    # if processed_memories['memories']:
    #     vectorizer = TfidfVectorizer(stop_words='english')
    #     vectorizer.fit(processed_memories['memories'])
    #     tfidf_matrix = compute_memory_embeddings(processed_memories['memories'], vectorizer)
    # else:
    #     print("No memories loaded.")

    print("=== Mira Dreaming Script is ready! ===")

    if args.mode == 'manual':
        print("Running in MANUAL mode. You can trigger dream tasks by pressing Enter.\n")
        print("Press 'Ctrl+C' to exit.\n")
        try:
            while True:
                input("Press Enter to trigger a dream task...")
                perform_dream_task(model, tokenizer, manual=True)
                print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print("\nScript terminated by user. Goodbye!")
    else:
        print("Running in AUTOMATIC mode. Dream tasks will be performed every 10 minutes.\n")
        print("Press 'Ctrl+C' to stop.\n")
        try:
            while True:
                start_time = time.time()
                perform_dream_task(model, tokenizer, manual=False)
                # Calculate elapsed time and sleep for the remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(0, DREAM_INTERVAL_SECONDS - elapsed)
                print(f"Sleeping for {sleep_time/60:.2f} minutes...\n{'-' * 50}\n")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nScript terminated by user. Goodbye!")

if __name__ == "__main__":
    main()
