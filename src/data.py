import torch

def generate_repeated_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pattern_len: int,
    device: str = "cpu"
):
    # Generate Sequences like: [a,b,c,d,a,b,c,d,...]

    assert seq_len % pattern_len == 0

    repeats = seq_len // pattern_len

    patterns = torch.randint(
        low = 0,
        high = vocab_size,
        size=(batch_size, pattern_len),
        device=device,
    )

    batch = patterns.repeat(1, repeats)

    return batch
    
def make_fixed_eval_set(
    num_examples: int,
    seq_len: int,
    vocab_size: int,
    pattern_len: int,
    device: str = "cpu"  

):
    #fixed eval set reused across checkpoints

    return generate_repeated_batch(
        batch_size=num_examples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern_len=pattern_len,
        device=device,
    )

def make_inputs_and_targets(batch):
    #For next token prediction
    #Example: batch: [A,B,C,D] inputs: [A,B,C] targets: [B,C,D]

    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets
