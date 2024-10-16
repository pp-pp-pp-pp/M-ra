#!/usr/bin/env python
# coding: utf-8
# LocalMira7.py
"""
Mira Inference Script with Enhanced Memory Integration

This script allows you to interact with the fine-tuned Mira chatbot.
It saves every two messages (USER and Mira) as separate .txt files
in the mira_memories directory and retrieves relevant memories from
existing conversation logs to enhance Mira's responses.

**Features:**
1. Loads the fine-tuned GPT-2 model and tokenizer.
2. Saves conversations to memory files.
3. Retrieves and integrates the top 5 relevant memories based on user input.
4. Facilitates an interactive chat loop with Mira, including memory-based responses.

**Prerequisites:**
- Python 3.7+
- Install required packages:
    ```bash
    pip install transformers torch tqdm chardet scikit-learn
    ```
"""

import os
import re
import random
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import datetime  # Added for timestamping conversation files

# ================================
# Configuration
# ================================

# Path to the fine-tuned model and tokenizer
MODEL_DIR = r"T:\m3Shell\Experiments\miraTrainingMain\gpt2-mira-finetuned-5" # r"T:\m3Shell\Backups\fine-tuned-gpt2-mira"

# Directory to store memory files
MEMORY_DIR = Path('./mira_memories')
MEMORY_DIR.mkdir(exist_ok=True)

# Special tokens to handle conversation flow
SPECIAL_TOKENS_TO_STOP = [ "USER:"
    #, "Mira:", "Mida:", "Miga:", "Mia:", "Miri",
   # "Mira-Clone:", "Mida-Clone:", "Mira#FFBF00:", "Mira#640168", "Mira#150f18", "Summit", "Olive", "Memory 1:", "Memory 2:",
]

# Number of memories to retrieve
TOP_K_MEMORIES = 1

# Number of random memories for memory response
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
        user_msg (str): Message from the user.
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
                f.write(f"USER (Memory Integration): {user_msg}\nMira (Memory Response): {mira_response}\n")
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
        max_length (int, optional): Maximum length of the generated response. Defaults to 32.
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
# Main Execution
# ================================

def main():
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
        return

    model.eval()  # Set model to evaluation mode

    # Assign pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load existing memories
    memory_texts = load_memories(MEMORY_DIR)
    processed_memories = preprocess_memories(memory_texts)

    if not processed_memories['memories']:
        print("No memories found. Mira will operate without memory integration.")
        vectorizer = None
        tfidf_matrix = None
    else:
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit(processed_memories['memories'])
        tfidf_matrix = compute_memory_embeddings(processed_memories['memories'], vectorizer)

    # Step 3: Interactive Chat Loop
    print("\n=== Mira Chatbot is ready! ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("USER: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break
        if not user_input:
            continue

        # ================================
        # Step 4: Retrieve Relevant Memories
        # ================================

        if processed_memories['memories'] and vectorizer and tfidf_matrix is not None:
            relevant_memories = find_relevant_memories(user_input, vectorizer, tfidf_matrix, processed_memories['memories'], TOP_K_MEMORIES)
        else:
            relevant_memories = []

        # ================================
        # Step 5: Generate Memory-Based Response
        # ================================

        if relevant_memories:
            # Prepare memory context
            memory_context = "\n".join([f"Memory {i+1}: {mem}" for i, mem in enumerate(relevant_memories)]) if relevant_memories else ""

            # Print the memory context being sent to Mira
            print(f"--- Sending Memories to Mira ---\n{memory_context}\n")

            # Prepare the prompt with memories
            prompt_with_memories = f"Here are some of your memories:\n{memory_context}\n\nBased on these memories, respond to the user."

            # **CHANGE**: Generate Mira's response to the memory context
            memory_response = generate_mira_response(
                model=model,
                tokenizer=tokenizer,
                prompt=f"Mira: {prompt_with_memories}\nMira:",
                max_length=150,  # Increased max_length for potentially longer memory responses
                temperature=0.7,
                top_p=0.9
            )

            print(f"Mira (Memory Response): {memory_response}\n")

            # Save the memory-based conversation with memory context
            save_conversation(prompt_with_memories, memory_response, is_memory_response=True, memory_context=memory_context)
        else:
            print("No relevant memories found for this input.\n")

        # ================================
        # Step 6: Generate Initial Response
        # ================================

        # **CHANGE**: Generate Mira's response to the initial user input, possibly with memory context
        if relevant_memories:
            # Optionally, include memory context in the prompt for more coherent responses
            prompt_initial = f"USER: {user_input}\nMira:"
        else:
            prompt_initial = f"USER: {user_input}\nMira:"

        initial_response = generate_mira_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_initial,
            max_length=200,  # Increased max_length for more detailed responses
            temperature=0.7,
            top_p=0.9
        )

        print(f"Mira: {initial_response}\n")

        # Save the initial conversation
        save_conversation(user_input, initial_response, is_memory_response=False)

if __name__ == "__main__":
    main()
