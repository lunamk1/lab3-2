import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    
    Args:
        input_ids (Tensor): Tensor of input token IDs (batch_size, seq_len).
        vocab_size (int): Size of the vocabulary.
        mask_token_id (int): ID used for [MASK] token.
        pad_token_id (int): ID used for [PAD] token.
        mlm_prob (float): Probability of masking a token. Default 0.15.
    
    Returns:
        input_ids_masked (Tensor): Masked input IDs.
        labels (Tensor): MLM labels (-100 for non-masked tokens to ignore loss).
    """
    labels = input_ids.clone()

    # Create a mask selecting tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices] = -100  # only compute loss on masked tokens (ignore non-masked)

    # 80% of the time: replace masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10% of the time: replace masked tokens with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    # Remaining 10%: keep original token

    return input_ids, labels

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
    """
    Train the encoder using masked language modeling.
    
    Args:
        model (nn.Module): The encoder model.
        dataloader (DataLoader): DataLoader for training data.
        tokenizer (Tokenizer): Tokenizer (for special token IDs).
        epochs (int, optional): Number of epochs. Defaults to 3.
        lr (float, optional): Learning rate. Defaults to 5e-4.
        device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    model.train()

    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Prepare masked inputs and labels
            masked_input_ids, labels = mask_tokens(input_ids.clone(), vocab_size, mask_token_id, pad_token_id)

            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            hidden_states = model(masked_input_ids, token_type_ids, attention_mask)
            logits = model.mlm_head(hidden_states)

            # Reshape logits and labels for loss computation
            logits = logits.view(-1, logits.size(-1))   # [batch_size * seq_len, vocab_size]
            labels = labels.view(-1)                    # [batch_size * seq_len]

            loss = loss_fn(logits, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Plot the loss curve
    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.show()