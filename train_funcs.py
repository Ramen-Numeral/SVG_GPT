# %%
import torch
import numpy as np
import time
from transformers import get_cosine_schedule_with_warmup
import os
import global_config as cfg
from model_design import GPT
import mup
import pickle


# quick param counter, mostly for logging + scaling law plots
def count_params(model):
    return sum(p.numel() for p in model.parameters())


# batch sampler (karpathy-style, just adapted for mmap + numpy input)
def get_batch(data, batch_size=cfg.BATCH_SIZE, block_size=cfg.BLOCK_SIZE, device=cfg.DEVICE):
    # random start indices for sequence chunks
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x is context, y is next-token shift
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix
    ])

    # pin memory helps cuda transfer overlap with compute
    if 'cuda' in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, val, device,
                  batch_size=cfg.BATCH_SIZE,
                  block_size=cfg.BLOCK_SIZE,
                  eval_iters=cfg.VAL_LOSS_ITERATIONS):
    """
    monte carlo estimate of validation loss
    (karpathy-style random window sampling instead of full pass)
    """

    model.eval()
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        x, y = get_batch(val, batch_size=batch_size, block_size=block_size, device=device)
        _, loss = model(x, y)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item(), losses.numpy()


# optimizer builder with mup-aware switching
def get_optimizer(model, lr, weight_decay=0.1, beta1=0.9, beta2=0.95):
    """
    splits params into decay / no-decay groups like karpathy code
    plus optional mup optimizer swap
    """

    use_mup = getattr(model.config, "use_mup", False)

    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # standard transformer trick:
    # weights decay for matmuls, not for layernorm / bias terms
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if use_mup:
        # mup replaces adamw with scaled parameterization aware version
        print(f"using mup adamw | lr={lr}")
        return mup.MuAdamW(optim_groups, lr=lr, betas=(beta1, beta2))

    print(f"using standard adamw | lr={lr}")
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(beta1, beta2))


def lr_sweep(config, data, device=cfg.DEVICE):
    """
    brute force lr search over logspace grid
    used to find stable training region per model size
    """

    results = []
    use_mup = getattr(config, "use_mup", False)

    for lr in cfg.LEARNING_RATES:
        model = create_model(config)
        optimizer = get_optimizer(model, lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.WARMUP_STEPS,
            num_training_steps=cfg.LR_SWEEP_STEPS
        )

        stats = train_model(model, data, optimizer, scheduler,
                            cfg.LR_SWEEP_STEPS, device)

        results.append((lr, stats["loss_curve"], stats["avg_loss"]))

        # cleanup between runs (important for gpu memory stability)
        del model, optimizer, scheduler

    tag = "mup" if use_mup else "standard"
    path = os.path.join(cfg.SAVE_DIR, f"{tag}_results.pkl")

    with open(path, 'wb') as f:
        pickle.dump(results, f)

    # pick best lr by lowest avg loss
    best_lr = min(results, key=lambda x: x[2])[0]
    return results, best_lr


def train_model(model, data, optimizer, scheduler, steps, device="cuda"):
    """
    core training loop

    mostly karpathy-style but:
    - adds token counting
    - tracks throughput + memory
    """

    # fallback safety for mac / cpu runs
    if not torch.cuda.is_available():
        device = "cpu"

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    model.train()

    losses = []
    total_tokens = 0
    token_count_arr = []
    start_time = time.time()

    for step in range(steps):
        x, y = get_batch(data,
                         batch_size=cfg.BATCH_SIZE,
                         block_size=cfg.BLOCK_SIZE,
                         device=device)

        total_tokens += x.numel()
        token_count_arr.append(total_tokens)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # karpathy-style gradient clipping (keeps training stable for deep stacks)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    duration = time.time() - start_time

    mem_usage = 0
    if device.startswith("cuda"):
        mem_usage = torch.cuda.max_memory_allocated() / 1e6

    return {
        "avg_loss": sum(losses) / len(losses),
        "loss_curve": losses,
        "training_duration": duration,
        "tokens_per_sec": total_tokens / duration,
        "mem_mb": mem_usage,
        "tokens_by_step": token_count_arr,
    }


def run_model_stage(name, train_data, val_data, config, lib, lr,
                    file_pref="run", device=cfg.DEVICE, train_bestest=False):
    """
    full training + logging pipeline for a single model config

    handles:
    - training loop
    - validation
    - logging into GPTLibrary
    - checkpointing
    """

    use_mup = getattr(config, "use_mup", False)

    model = create_model(config)
    params = count_params(model)

    # either full training or single epoch run
    epochs = cfg.EPOCHS if train_bestest else 1

    total_tokens = len(train_data)
    epoch_steps = total_tokens // (cfg.BATCH_SIZE * cfg.BLOCK_SIZE)

    optimizer = get_optimizer(model, lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.WARMUP_STEPS,
        num_training_steps=epoch_steps * epochs
    )

    print(f"training {name} | params={params:,} | lr={lr}")

    for i in range(epochs):

        stats = train_model(model, train_data, optimizer, scheduler, epoch_steps)

        val_loss, val_curve = estimate_loss(model, val_data, device)

        # rename epochs if doing full sweep runs
        run_name = f'{name}_epoch_{i}' if train_bestest else name

        # logging scalars
        lib.log_summary(run_name, 'params', params)
        lib.log_summary(run_name, 'mean_val_loss', val_loss)
        lib.log_summary(run_name, 'mean_tr_loss', stats["avg_loss"])
        lib.log_summary(run_name, 'tok_per_sec', stats["tokens_per_sec"])
        lib.log_summary(run_name, 'train_time', stats["training_duration"])
        lib.log_summary(run_name, 'mem_mb', stats["mem_mb"])
        lib.log_summary(run_name, 'lr', lr)

        # time-series logging (loss curves for scaling law plots)
        lib.log_series(run_name, stats, val_results=val_curve)

        # checkpoint per epoch (simple but heavy disk usage)
        ckpt_path = os.path.join(cfg.SAVE_DIR, f"{file_pref}_{run_name}_epoch{i}.pt")
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "val_loss": val_loss,
            "params": params
        }, ckpt_path)

        print(f"saved checkpoint: {ckpt_path}")

    lib.print_mod_stats(name)

    del model, optimizer, scheduler
    return


def create_model(config):
    """
    builds model + applies mup alignment if needed

    mup part:
    - base model defines reference parameterization
    - delta model helps define width scaling behavior
    """

    model = GPT(config).to(cfg.DEVICE)

    if config.use_mup:
        with torch.no_grad():
            base_model = GPT(cfg.BASE_CONFIG).to(cfg.DEVICE)
            delta_model = GPT(cfg.DELTA_CONFIG).to(cfg.DEVICE)

            # aligns init so scaling laws behave correctly across widths
            mup.set_base_shapes(model, base_model, delta=delta_model)

            del base_model, delta_model

    model.apply(model._init_weights)
    return model
