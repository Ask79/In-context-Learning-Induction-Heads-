import json
import torch
from torch.optim import AdamW

from main import TinyTransformerLM
from data import generate_repeated_batch, make_fixed_eval_set, make_inputs_and_targets

def evaluate(model, eval_set, batch_size, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0

    losses_by_pos = None

    
    with torch.no_grad():
        for i in range(0, eval_set.size(0), batch_size):
            batch = eval_set[i:i + batch_size].to(device)
            inputs, targets = make_inputs_and_targets(batch)
            

            logits, loss = model(inputs, targets)

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

            per_token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction ="none",

            ).view(targets.shape)

            pos_loss = per_token_loss.mean(dim=0)

            if losses_by_pos is None:
                losses_by_pos = pos_loss
            else:
                losses_by_pos += pos_loss

    losses_by_pos = losses_by_pos / (eval_set.size(0) / batch_size)

    early_pos = 20
    late_pos = 100
    icl_score = losses_by_pos[early_pos].item() - losses_by_pos[late_pos].item()

    

    model.train()

    return {
        "eval_loss": total_loss / total_tokens,
        "icl_score": icl_score,
        "loss_early": losses_by_pos[early_pos].item(),
        "loss_late": losses_by_pos[late_pos].item(),

    }

def train_model(
        n_layers: int,
        steps: int = 1000, 
        eval_every: int = 50,
        vocab_size: int = 128,
        seq_len: int = 128,
        pattern_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        batch_size: int = 64,
        eval_examples: int = 1024,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    
):
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len - 1,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        use_mlp=False,

    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    eval_set = make_fixed_eval_set(
        num_examples=eval_examples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern_len=pattern_len,
        device="cpu",
    )

    print("Example eval sequence:")
    print(eval_set[0])

    metrics = []

    for step in range(1, steps+1):
        batch = generate_repeated_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            pattern_len=pattern_len,
            device=device,
        )

        inputs, targets = make_inputs_and_targets(batch)

        logits, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            eval_metrics = evaluate(model, eval_set, batch_size, device)

            row = {
                "step": step,
                "n_layers": n_layers,
                "train_loss": loss.item(),
                **eval_metrics,
            }

            metrics.append(row)
            print(
                f"Step {step} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Eval Loss: {eval_metrics['eval_loss']:.4f} | "
                f"ICL: {eval_metrics['icl_score']:.4f}"
            )
            print(row)
    model.eval()

    batch = eval_set[:1].to(device)
    inputs, targets = make_inputs_and_targets(batch)

    with torch.no_grad():
        logits, _ = model(inputs)

    preds = logits.argmax(dim=-1)

    print("Example inputs:")
    print(inputs[0][:80])

    print("Example targets:")
    print(targets[0][:80])

    print("Example predictions:")
    print(preds[0][:80])
    
    query_pos = 101
    print("\nToken inspection:")
    for pos in [4, 5, 36, 37, 68, 69, 100]:
        print(
            f"pos {pos} | "
            f"input={inputs[0, pos].item()} | "
            f"target={targets[0, pos].item()}"
        )
    print("\nRepeated answer positions:")
    for pos in [5, 37, 69]:
        print(
            f"pos {pos} | "
            f"input={inputs[0, pos].item()} | "
            f"target={targets[0, pos].item()}"
        )   

    with torch.no_grad():
        logits, loss, all_attn = model(
            inputs,
            targets,
            return_attn=True,
        )

    print("Number of layers:", len(all_attn))
    print("Attention shape:", all_attn[0].shape)

    print("First attention head:")
    print(all_attn[0][0, 0, :10, :10])

    query_pos = 101

    for layer_idx, attn in enumerate(all_attn): 
        # attn shape: (batch, heads, query_pos, key_pos)
        print(f"\nLayer {layer_idx} attention at query position {query_pos}:")

        for head_idx in range(attn.shape[1]):
            weights = attn[0, head_idx, query_pos]

            top_vals, top_idxs = torch.topk(weights, k=10)

            print(f"Head {head_idx}:")
            for val, idx in zip(top_vals, top_idxs):
                idx = idx.item()
                val = val.item()
                distance = query_pos - idx
                print(f"  attends to pos {idx:3d} | distance back {distance:3d} | weight {val:.4f}")

    output_path = f"metrics_{n_layers}layer.json"
    with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    train_model(n_layers=1)
    train_model(n_layers=2)

