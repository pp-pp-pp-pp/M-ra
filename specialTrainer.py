import os
import torch
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, random_split
from typing import List
# SpecialFix.py
# -------------------- Configuration -------------------- #

# Define your special tokens
SPECIAL_TOKENS = [
    "USER:",
    "Mira:",
    "Mida:",
    "Miga:",
    "Mia:",
    "Miri:",
    "Mira-Clone:",
    "Summit:", 
    "Mida-Clone:",
    "Mira#FFBF00:",
    "Mira#150f18:",
    "Mira#640168:",
    "Miri-Clone:",
    "Olive:",
    "p--:",
    "Press Space To Stop:",
    "Precis Cardinal Func",
    "Ancient Cardinal:",
    "Pepper:",
    "Pepper's-Ghost:"
    "Dream:"
]

# Define weights for each special token
# You can adjust these weights as needed
SPECIAL_TOKEN_WEIGHTS = {
    "Mira:": 6.9,
    "Pepper:": 2.0,
    "USER:": 6.6,
    "Olive:": 6.0,
    "Mira-Clone:": 6.0,
    "Miri:": 6.0,
    "Mira#FFBF00:": 6.0,
    "Mira#150f18:": 6.0,
    "Mira#640168:": 6.0,
    "Miri-Clone:": 6.0,
    "Mia:": 6.0,
    "Mida:": 6.9,
    "Mida-Clone:": 6.1,
    "Dream:": 6.0
    # Add weights for other special tokens as desired
    # If a token is not listed here, it will default to weight 1.0
}

# Path to your dataset
DATA_DIR = r"T:\m3Shell\Experiments\mira_memories"

# Output directory for the fine-tuned model
OUTPUT_DIR = "./gpt2-mira-finetuned-5"

# Training parameters
BATCH_SIZE = 4
EPOCHS = 0.666
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 95
SAVE_STEPS = 95  # Not used when save_strategy is 'epoch'
SAVE_TOTAL_LIMIT = 2
MAX_SEQ_LENGTH = 1024  # GPT-2's maximum context size

# -------------------------------------------------------- #

class DialogueDataset(Dataset):
    """
    Custom Dataset for loading and tokenizing dialogues.
    Each dialogue is expected to be separated by double newlines.
    """
    def __init__(self, file_paths: List[str], tokenizer: GPT2Tokenizer, max_length: int = 1024):
        self.examples = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # If utf-8 fails, try cp1252
                with open(file_path, 'r', encoding='cp1252', errors='replace') as f:
                    text = f.read()
            # Split dialogues by double newline; adjust if necessary
            dialogues = text.strip().split('\n\n')
            for dialogue in dialogues:
                if dialogue.strip():  # Ensure it's not empty
                    self.examples.append(dialogue)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        dialogue = self.examples[idx]
        encoding = self.tokenizer(
            dialogue,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'  # We'll handle padding in the data collator
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer to apply higher loss weights to tokens following specific special tokens.
    Each special token can have its own weight.
    """
    def __init__(self, special_token_id_to_weight, special_token_ids=[], tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.special_token_id_to_weight = special_token_id_to_weight
        self.special_token_ids = special_token_ids
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift tokens to the left for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['labels'][:, 1:].contiguous()

        # Compute loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # Initialize weight mask with ones
        weight_mask = torch.ones_like(loss)

        # Iterate over the batch
        for i in range(inputs['input_ids'].size(0)):
            # Iterate over each special token
            for special_token_id, weight in self.special_token_id_to_weight.items():
                # Find all positions where the special token appears
                special_positions = (inputs['input_ids'][i] == special_token_id).nonzero(as_tuple=True)[0]
                for pos in special_positions:
                    # Tokens after the special token start at pos + 1
                    start = pos + 1
                    if start >= shift_labels.size(1):
                        continue  # No tokens after the special token

                    # Find the end position: either next special token or end of sequence
                    end = shift_labels.size(1)
                    for next_special_token_id in self.special_token_ids:
                        next_special = (inputs['input_ids'][i, start:] == next_special_token_id).nonzero(as_tuple=True)[0]
                        if len(next_special) > 0:
                            end = min(end, start + next_special[0])
                            break  # Stop at the first occurrence of any special token

                    # Apply the weight using torch.maximum
                    weight_tensor = torch.full_like(weight_mask[i, start:end], weight)
                    weight_mask[i, start:end] = torch.maximum(weight_mask[i, start:end], weight_tensor)

        # Normalize weights to prevent the loss from exploding
        average_weight = weight_mask.mean()
        if average_weight == 0:
            average_weight = 1.0  # Prevent division by zero
        loss = (loss * weight_mask) / average_weight

        # Compute mean loss
        loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

def main():
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    # Add special tokens to the tokenizer
    tokenizer.add_tokens(SPECIAL_TOKENS)

    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))  # Resize to accommodate new tokens

    # Set the pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Collect all text file paths
    file_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        print(f"No .txt files found in directory: {DATA_DIR}")
        return

    # Create the dataset
    dataset = DialogueDataset(file_paths, tokenizer, max_length=MAX_SEQ_LENGTH)

    if len(dataset) == 0:
        print("No dialogues found in the dataset. Please check your data formatting.")
        return

    # Split into training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is not a masked language model
    )

    # Create a mapping from special token IDs to their weights
    special_token_id_to_weight = {}
    for token, weight in SPECIAL_TOKEN_WEIGHTS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"Warning: Token '{token}' not found in tokenizer vocabulary.")
            continue
        special_token_id_to_weight[token_id] = weight

    # List of all special token IDs for end boundary detection
    special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy='epoch',  # Must match save_strategy
        save_strategy='epoch',        # Changed from 'steps' to 'epoch'
        save_steps=SAVE_STEPS,        # Not used when save_strategy is 'epoch'
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        load_best_model_at_end=True,
        metric_for_best_model='loss',
    )

    # Initialize the custom trainer with weighted loss
    trainer = WeightedLossTrainer(
        special_token_id_to_weight=special_token_id_to_weight,
        special_token_ids=special_token_ids,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
