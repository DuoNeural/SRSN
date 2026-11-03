import os
import sys
import time
import math
import psutil
import resource
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from tokenizers import Tokenizer

# ==============================================================================
# 0. HARDWARE THREAD PINNING (AMD RYZEN 5 3550H: 4 PHYSICAL CORES)
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PATH"] = f"/home/duoneural/srsn_env/bin:{os.environ.get('PATH', '')}"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0

def get_current_rss_mb():
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

# ==============================================================================
# 1. LOAD NATIVE C++ OPENMP / AVX2 SIMD KERNEL
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_src = os.path.join(current_dir, "sdoc_kernel.cpp")

sdoc_cpp = load(
    name="sdoc_cpp",
    sources=[cpp_src],
    extra_cflags=["-O3", "-march=native", "-mavx2", "-mfma", "-fopenmp", "-ffast-math"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)

class SDOCCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_ext, gamma, omega_sq, alpha, dt: float):
        outputs, q_hist, p_hist = sdoc_cpp.forward(F_ext, gamma, omega_sq, alpha, float(dt))
        ctx.save_for_backward(gamma, q_hist, p_hist, omega_sq, alpha)
        ctx.dt = dt
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        gamma, q_hist, p_hist, omega_sq, alpha = ctx.saved_tensors
        dF_ext, dgamma, domega_sq, dalpha = sdoc_cpp.backward(
            grad_out, gamma, q_hist, p_hist, omega_sq, alpha, float(ctx.dt)
        )
        return dF_ext, dgamma, domega_sq, dalpha, None

# ==============================================================================
# 2. ZERO-COPY UINT16 MEMMAP DATASTREAMER
# ==============================================================================
class MemmapDataset:
    def __init__(self, bin_path: str, seq_len: int = 1024):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}. Run prepare_data.py first!")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.num_tokens = len(self.data)
        if self.num_tokens <= seq_len + 1:
            raise ValueError(f"Dataset in {bin_path} has only {self.num_tokens} tokens, need > {seq_len + 1}")

    def sample_batch(self, batch_size: int, device: torch.device):
        ix = np.random.randint(0, self.num_tokens - self.seq_len - 1, size=batch_size)
        batch = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
        t_batch = torch.from_numpy(batch).to(device)
        return t_batch[:, :-1].contiguous(), t_batch[:, 1:].contiguous()

# ==============================================================================
# 3. OPTIMIZER: NEWTON-SCHULZ ORTHOGONALIZED MOMENTUM (MUON)
# ==============================================================================
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    norm = X.norm() + eps
    X = X / norm
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = torch.mm(X, X.T)
        B = b * A + c * torch.mm(A, A)
        X = a * X + torch.mm(B, X)
    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(f"Muon only supports 2D matrices. Got {p.shape}")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                u = g + momentum * buf if nesterov else buf
                u_ortho = zeropower_via_newtonschulz5(u)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(u_ortho, alpha=-lr * scale)

# ==============================================================================
# 4. NEURAL ARCHITECTURE COMPONENTS
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SDOC(nn.Module):
    def __init__(self, d_state: int, n_layers: int = 12, dt: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.dt = dt
        self.norm = RMSNorm(d_state)
        self.w_force = nn.Linear(d_state, d_state, bias=False)
        self.w_damp = nn.Linear(d_state, d_state, bias=True)
        self.w_out = nn.Linear(d_state, d_state, bias=False)

        nn.init.normal_(self.w_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        taus = torch.exp(torch.linspace(math.log(2.0), math.log(2048.0), d_state))
        self.omega_sq = nn.Parameter((2.0 * math.pi / taus) ** 2)
        self.alpha = nn.Parameter(torch.ones(d_state) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        F_ext = self.w_force(x_norm)
        # Guarantee strictly bounded non-zero dissipation to prevent numerical energy accumulation
        gamma = torch.clamp(torch.sigmoid(self.w_damp(x_norm)), min=0.01, max=0.99)
        # Bounded stiffness spectrum protecting against high-frequency resonance singularities
        omega_sq = torch.clamp(torch.abs(self.omega_sq), min=1e-4, max=100.0)
        y = SDOCCppFunction.apply(F_ext, gamma, omega_sq, self.alpha, self.dt)
        return self.w_out(y)

class SwiGLU(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 12):
        super().__init__()
        self.norm = RMSNorm(d_state)
        self.w_gate = nn.Linear(d_state, d_ff, bias=False)
        self.w_up = nn.Linear(d_state, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_state, bias=False)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.w_down(F.silu(self.w_gate(x_norm)) * self.w_up(x_norm))

class ReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y1: torch.Tensor, y2: torch.Tensor, f_module: nn.Module, g_module: nn.Module):
        ctx.f_module = f_module
        ctx.g_module = g_module
        with torch.no_grad():
            f_out = f_module(y2)
            y1_next = y1 + f_out
            g_out = g_module(y1_next)
            y2_next = y2 + g_out
        ctx.save_for_backward(y1_next.detach(), y2_next.detach())
        return y1_next, y2_next

    @staticmethod
    def backward(ctx, dy1_next: torch.Tensor, dy2_next: torch.Tensor):
        y1_next, y2_next = ctx.saved_tensors
        f_module = ctx.f_module
        g_module = ctx.g_module

        # Step 1 & 2: Reconstruct y2 and backprop through G in one pass
        with torch.enable_grad():
            y1_next_eval = y1_next.detach().requires_grad_(True)
            g_eval = g_module(y1_next_eval)
            g_eval.backward(dy2_next)
            dy1_from_g = y1_next_eval.grad

        y2 = y2_next - g_eval.detach()
        dy1 = dy1_next + dy1_from_g

        # Step 3 & 4: Reconstruct y1 and backprop through F in one pass
        with torch.enable_grad():
            y2_eval = y2.detach().requires_grad_(True)
            f_eval = f_module(y2_eval)
            f_eval.backward(dy1)
            dy2_from_f = y2_eval.grad

        y1 = y1_next - f_eval.detach()
        dy2 = dy2_next + dy2_from_f
        return dy1, dy2, None, None

class ReversibleBlock(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 12):
        super().__init__()
        self.f_module = SDOC(d_state=d_state, n_layers=n_layers)
        self.g_module = SwiGLU(d_state=d_state, d_ff=d_ff, n_layers=n_layers)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        return ReversibleBlockFunction.apply(y1, y2, self.f_module, self.g_module)

class SRSNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16384, d_model: int = 640, n_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_model // 2
        self.d_ff = 856

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            ReversibleBlock(d_state=self.d_state, d_ff=self.d_ff, n_layers=n_layers)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        y1, y2 = torch.chunk(x, chunks=2, dim=-1)
        for layer in self.layers:
            y1, y2 = layer(y1, y2)
        out = torch.cat([y1, y2], dim=-1)
        return self.final_norm(out)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, chunk_tokens: int = 512):
        out = self.forward_features(input_ids)
        if targets is None:
            return F.linear(out, self.embedding.weight)

        flat_out = out.view(-1, self.d_model)
        flat_targets = targets.view(-1)
        total_tokens = flat_out.size(0)
        total_loss = 0.0
        for start in range(0, total_tokens, chunk_tokens):
            end = min(start + chunk_tokens, total_tokens)
            logits_chunk = F.linear(flat_out[start:end], self.embedding.weight)
            loss_chunk = F.cross_entropy(logits_chunk, flat_targets[start:end], reduction='sum')
            total_loss = total_loss + loss_chunk
        return total_loss / total_tokens

# ==============================================================================
# 5. AUTOREGRESSIVE GENERATION
# ==============================================================================
@torch.no_grad()
def generate(model: SRSNLanguageModel, tokenizer, prompt: str, max_new_tokens: int = 48, temperature: float = 0.8, top_k: int = 40):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    for _ in range(max_new_tokens):
        # Truncate context if exceeding 1024
        curr_input = input_tensor if input_tensor.size(1) <= 1024 else input_tensor[:, -1024:]
        logits = model(curr_input)
        next_token_logits = logits[:, -1, :] / max(1e-5, temperature)
        
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        
        # Stop at EOS
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break
            
    model.train()
    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

# ==============================================================================
# 6. PRODUCTION TRAINING LOOP
# ==============================================================================
def get_lr_factor(step, warmup_steps, total_steps, start_step=0, resume_warmup_steps=0, min_ratio=0.1):
    # Smooth re-warmup when resuming from an already decayed checkpoint to avoid 10x impulse shock
    if start_step > warmup_steps and resume_warmup_steps > 0:
        if step <= start_step + resume_warmup_steps:
            target_progress = float(start_step + resume_warmup_steps - warmup_steps) / float(max(1, total_steps - warmup_steps))
            target_cosine = 0.5 * (1.0 + math.cos(math.pi * target_progress))
            target_scale = min_ratio + (1.0 - min_ratio) * target_cosine
            alpha = float(step - start_step) / float(max(1, resume_warmup_steps))
            return min_ratio + alpha * (target_scale - min_ratio)

    if step < warmup_steps:
        return max(1e-4, float(step) / float(max(1, warmup_steps)))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

def train_production():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/duoneural/srsn_ryzen")
    p.add_argument("--train-bin", default=None, help="Path to train.bin (defaults to /dev/shm or data-dir)")
    p.add_argument("--val-bin", default=None, help="Path to val.bin (defaults to /dev/shm or data-dir)")
    p.add_argument("--resume-checkpoint", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--total-optimizer-steps", type=int, default=3250)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--resume-warmup-steps", type=int, default=25, help="Gradual re-warmup steps when resuming from decayed checkpoint")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--eval-interval", type=int, default=25)
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"\n==============================================================================", flush=True)
    print(f"[Init] SRSN Production Pre-Training (AMD Ryzen 5 3550H / Zen+ AVX2)", flush=True)
    print(f"       Batch={args.batch_size}, SeqLen={args.seq_len}, Accum={args.grad_accum_steps}", flush=True)
    print(f"       Tokens/OptStep: {args.batch_size * args.seq_len * args.grad_accum_steps:,}", flush=True)
    print(f"==============================================================================", flush=True)

    tok_path = os.path.join(args.data_dir, "tokenizer_16k.json")
    
    # Resolve train/val dataset paths (prefer /dev/shm RAM disk)
    if args.train_bin:
        train_bin = args.train_bin
    elif os.path.exists("/dev/shm/train_golden.bin"):
        train_bin = "/dev/shm/train_golden.bin"
    elif os.path.exists("/dev/shm/train.bin"):
        train_bin = "/dev/shm/train.bin"
    elif os.path.exists(os.path.join(args.data_dir, "train_golden.bin")):
        train_bin = os.path.join(args.data_dir, "train_golden.bin")
    else:
        train_bin = os.path.join(args.data_dir, "train.bin")

    if args.val_bin:
        val_bin = args.val_bin
    elif os.path.exists("/dev/shm/val_golden.bin"):
        val_bin = "/dev/shm/val_golden.bin"
    elif os.path.exists("/dev/shm/val.bin"):
        val_bin = "/dev/shm/val.bin"
    elif os.path.exists(os.path.join(args.data_dir, "val_golden.bin")):
        val_bin = os.path.join(args.data_dir, "val_golden.bin")
    else:
        val_bin = os.path.join(args.data_dir, "val.bin")

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}! Run prepare_data.py first.")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    print(f"[Data] Training Dataset:   {train_bin} ({os.path.getsize(train_bin)/(1024*1024):.2f} MB)")
    print(f"[Data] Validation Dataset: {val_bin} ({os.path.getsize(val_bin)/1024:.2f} KB)")
    train_ds = MemmapDataset(train_bin, seq_len=args.seq_len)
    val_ds = MemmapDataset(val_bin, seq_len=args.seq_len)
    print(f"[Data] Loaded memmap dataset: Train={train_ds.num_tokens:,} tokens | Val={val_ds.num_tokens:,} tokens", flush=True)

    model = SRSNLanguageModel(vocab_size=vocab_size, d_model=640, n_layers=12).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embedding.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    print(f"[Model] 16k Vocab SRSN Parameter Breakdown:")
    print(f"        - Embedding Table ({vocab_size} x 640): {embed_params / 1e6:.2f}M ({embed_params:,} params)")
    print(f"        - 12 RevNet Trunk Layers:            {layer_params / 1e6:.2f}M ({layer_params:,} params)")
    print(f"        - Total Tied Parameter Footprint:     {total_params / 1e6:.2f}M ({total_params:,} params)", flush=True)

    # Segregate parameter groups
    # 1. Muon for 2D trunk matrices
    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embedding" not in n]
    
    # 2. AdamW Decay: Embedding matrix
    adam_decay_params = [p for n, p in model.named_parameters() if "embedding" in n]
    
    # 3. AdamW No-Decay: omega_sq, alpha, RMSNorm weights, biases (crucial: protects resonance frequency bank!)
    adam_no_decay_params = [p for n, p in model.named_parameters() if p.ndim != 2 and "embedding" not in n]

    print(f"[Optim] Parameter Segregation:")
    print(f"        - Muon 2D Matrices (weight_decay=0.01):      {sum(p.numel() for p in muon_params) / 1e6:.2f}M ({len(muon_params)} tensors)")
    print(f"        - AdamW Decay (Embedding, wd=0.01):          {sum(p.numel() for p in adam_decay_params) / 1e6:.2f}M ({len(adam_decay_params)} tensors)")
    print(f"        - AdamW No-Decay (omega_sq, alpha, norms):   {sum(p.numel() for p in adam_no_decay_params):,} params ({len(adam_no_decay_params)} tensors)", flush=True)

    optimizer_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01)
    optimizer_adam = torch.optim.AdamW([
        {"params": adam_decay_params, "weight_decay": 0.01},
        {"params": adam_no_decay_params, "weight_decay": 0.0}
    ], lr=args.adam_lr, betas=(0.9, 0.95))

    ckpt_dir = os.path.join(args.data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_opt_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Resume] Loading checkpoint from {args.resume_checkpoint}...", flush=True)
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_muon" in ckpt:
            try:
                optimizer_muon.load_state_dict(ckpt["optimizer_muon"])
            except Exception as e:
                print(f"[Resume] Note: Muon optimizer state load skipped: {e}")
        if "optimizer_adam" in ckpt:
            try:
                optimizer_adam.load_state_dict(ckpt["optimizer_adam"])
            except Exception as e:
                print(f"[Resume] Note: AdamW optimizer state load skipped: {e}")
        start_opt_step = ckpt.get("opt_step", 0)
        print(f"[Resume] Resumed successfully from Step {start_opt_step} (Recorded Loss: {ckpt.get('loss', 0.0):.4f})", flush=True)

    opt_step = start_opt_step
    start_micro_step = start_opt_step * args.grad_accum_steps
    total_micro_steps = args.total_optimizer_steps * args.grad_accum_steps
    remaining_micro_steps = total_micro_steps - start_micro_step
    
    t0 = time.perf_counter()

    print(f"[Run] Launching Pretraining: Steps {start_opt_step + 1} -> {args.total_optimizer_steps} ({remaining_micro_steps} micro-steps remaining)...", flush=True)

    model.train()
    optimizer_muon.zero_grad()
    optimizer_adam.zero_grad()

    for m_step in range(start_micro_step + 1, total_micro_steps + 1):
        step_t0 = time.perf_counter()
        inputs, targets = train_ds.sample_batch(args.batch_size, device)

        loss = model(inputs, targets=targets)
        
        # Micro-step NaN Guard: Never accumulate non-finite loss into gradients
        if not torch.isfinite(loss):
            print(f"  [GUARD] Non-finite loss ({loss.item()}) at uStep {m_step:04d}! Skipping backward pass.", flush=True)
            continue

        (loss / args.grad_accum_steps).backward()

        step_dt = time.perf_counter() - step_t0
        tok_s = (args.batch_size * args.seq_len) / step_dt

        # Check for optimizer update
        if m_step % args.grad_accum_steps == 0:
            opt_step += 1
            lr_scale = get_lr_factor(opt_step, args.warmup_steps, args.total_optimizer_steps, start_step=start_opt_step, resume_warmup_steps=args.resume_warmup_steps)
            
            for g in optimizer_muon.param_groups:
                g['lr'] = args.muon_lr * lr_scale
            for g in optimizer_adam.param_groups:
                g['lr'] = args.adam_lr * lr_scale

            # Optimizer Step Guard: verify all gradients are finite
            grads_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                print(f"  [GUARD] Non-finite gradients detected at Step {opt_step:03d} (uStep {m_step:04d})! Skipping optimizer update and zeroing grads.", flush=True)
                optimizer_muon.zero_grad()
                optimizer_adam.zero_grad()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"  [GUARD] Non-finite GradNorm ({grad_norm.item()}) at Step {opt_step:03d}! Skipping optimizer update.", flush=True)
                optimizer_muon.zero_grad()
                optimizer_adam.zero_grad()
                continue

            optimizer_muon.step()
            optimizer_adam.step()
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()

            rss = get_current_rss_mb()
            print(f"Step {opt_step:03d}/{args.total_optimizer_steps:03d} (uStep {m_step:04d}) | Loss: {loss.item():.4f} | GradNorm: {grad_norm.item():.3f} | LR_scale: {lr_scale:.3f} | Speed: {tok_s:6.1f} tok/s | RSS: {rss:6.1f} MB", flush=True)

            # Periodic Validation & Generation
            if opt_step % args.eval_interval == 0 or opt_step == args.total_optimizer_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(4):
                        v_in, v_tgt = val_ds.sample_batch(args.batch_size, device)
                        v_loss = model(v_in, targets=v_tgt)
                        val_losses.append(v_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                model.train()

                sample_story = generate(model, tokenizer, prompt="Once upon a time, Lily", max_new_tokens=48, temperature=0.8)
                sample_reason = generate(model, tokenizer, prompt="User: Is a cat an animal?\n\nAssistant:", max_new_tokens=32, temperature=0.8)
                print(f"  --> [Validation] Val Loss: {avg_val_loss:.4f}", flush=True)
                print(f"  --> [Sample Story] \"{sample_story.strip()}\"", flush=True)
                print(f"  --> [Sample Reason] \"{sample_reason.strip()}\"", flush=True)

            # Atomic Checkpointing
            if opt_step % args.save_interval == 0 or opt_step == args.total_optimizer_steps:
                ckpt_path = os.path.join(ckpt_dir, "srsn_latest.pt")
                step_ckpt_path = os.path.join(ckpt_dir, f"srsn_step{opt_step:04d}.pt")
                state = {
                    "opt_step": opt_step,
                    "model_state": model.state_dict(),
                    "optimizer_muon": optimizer_muon.state_dict(),
                    "optimizer_adam": optimizer_adam.state_dict(),
                    "loss": loss.item(),
                    "vocab_size": vocab_size,
                    "d_model": model.d_model,
                    "n_layers": len(model.layers)
                }
                torch.save(state, ckpt_path)
                torch.save(state, step_ckpt_path)
                print(f"  --> [Checkpoint] Saved atomic checkpoint to {step_ckpt_path}", flush=True)

        else:
            print(f"  [Micro-step {m_step:04d}] Loss: {loss.item():.4f} | Speed: {tok_s:6.1f} tok/s", flush=True)

    total_wall = time.perf_counter() - t0
    total_tokens = remaining_micro_steps * args.batch_size * args.seq_len
    overall_speed = total_tokens / total_wall if total_wall > 0 else 0
    print(f"\n==============================================================================", flush=True)
    print(f"[Complete] Pre-training Verification Run Finished!", flush=True)
    print(f"           Total Tokens Processed: {total_tokens:,}", flush=True)
    print(f"           Total Wall Time:        {total_wall:.2f}s ({total_wall/60:.2f} mins)", flush=True)
    print(f"           Sustained Speed:        {overall_speed:.2f} tokens/sec", flush=True)
    print(f"           Peak RSS Memory:        {get_peak_rss_mb():.2f} MB", flush=True)
    print(f"==============================================================================", flush=True)

if __name__ == "__main__":
    train_production()
