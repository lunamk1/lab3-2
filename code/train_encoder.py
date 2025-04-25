import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob)
    special_tokens_mask = (input_ids == pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels

def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-4, device='cuda'):
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

            # ✅ 确保 attention_mask 是 [batch, seq_len] 且类型为 bool
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.bool()

            masked_input_ids, labels = mask_tokens(
                input_ids,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                mlm_prob=0.15
            )
            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            hidden_states = model(masked_input_ids, token_type_ids, attention_mask)
            logits = model.mlm_head(hidden_states)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    plt.plot(range(1, epochs+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.show()
