# Running Cheapscape on rented hardware

The laptop stays the development machine. Anything that needs a GPU runs on a
rented instance that we treat as disposable: it can vanish mid-step, and the run
must survive that.

## Where each stage runs

| Stage | Machine | Why |
| --- | --- | --- |
| Corpus download, normalize, dedupe | Laptop or a cheap CPU VM | I/O bound, no GPU helps |
| Tokenizer training | Laptop | Minutes on a small sample |
| Packing to shards | Laptop or CPU VM | CPU + disk bound; do it once per corpus/tokenizer version |
| Overfit gate (`configs/overfit.yaml`) | Laptop CPU | Correctness only; free |
| Throughput benchmark | Rented GPU, minutes | Needs the real device to be meaningful |
| Pretraining | Rented GPU, interruptible | The only stage that justifies real spend |
| Evaluation, sampling | Laptop or the same GPU | Small model, cheap either way |
| Post-training | Rented GPU, short run | Far fewer tokens than pretraining |

Never pay a GPU to wait on tokenization. Packed shards are inputs to the GPU
stage, not something it builds.

## Choosing a platform

Indicative on-demand rates from a July 2026 survey. Marketplace prices move
hourly, so check live rates before committing.

| Platform | Strength | Watch out for |
| --- | --- | --- |
| Vast.ai | Cheapest; per-second billing; interruptible tier is roughly half price | Host quality varies, so filter on reliability score and disk speed |
| RunPod Community | Cheap with less variance; per-second billing; ready PyTorch templates | Community hosts are still peer-provided |
| Lambda | Most predictable, simplest networking and storage | Highest hourly rate; little reason to pay it at this scale |

Representative rates: RTX 4090 24 GB about \$0.27–0.40/hr, L40S 48 GB about
\$0.53–0.79/hr, A100 80 GB about \$0.67–1.40/hr, H100 80 GB about \$0.90–2.90/hr.

**Recommendation for the \$100 budget: a single RTX 4090 (or L40S) on Vast.ai
interruptible, or RunPod Community.** The v0 model in `configs/model.yaml` is
about 12.6M parameters at a 1,024-token context, which fits a 24 GB card with
room for a large batch. An H100 would finish sooner but buys far fewer GPU-hours
per dollar, and this project is bounded by total tokens rather than wall-clock
urgency. Multi-GPU is not worth the complexity at this model size.

At \$0.35/hr, \$100 buys roughly 285 GPU-hours; at \$1.00/hr, roughly 100. Hold
back about 20% for benchmarking, restarts, and post-training.

## Before you rent

1. Pass the overfit gate locally: `python3 scripts/train.py --config configs/overfit.yaml`.
   A loss that will not collapse on one batch is a bug, and a GPU will not fix it.
2. Pack the corpus and know its token count.
3. Decide the token budget, then let the benchmark tell you what it costs.

## Runbook

```bash
# On a fresh Ubuntu GPU instance
git clone <repo> cheapscape && cd cheapscape
./deploy/bootstrap.sh                 # system deps, venv, install, CUDA check

# Confirm the instance is what you paid for, and price the run
python3 scripts/benchmark.py \
  --contexts 512 1024 --batch-size 8 --precision bf16 \
  --token-budget 500000000 --price-per-hour-usd 0.35

# Copy the packed shards up (or re-pack on the box from raw text)
# then set price_per_hour_usd and budget_usd in your run config and launch
./deploy/run_pretrain.sh configs/train.yaml
```

`run_pretrain.sh` relaunches training when the process exits 75, which is what
`scripts/train.py` returns after a preemption signal. Training resumes from the
newest checkpoint, so a reclaimed instance costs at most one checkpoint interval
of work.

## The preemption contract

- `SIGTERM` or `SIGINT` sets a flag; the loop finishes the current step, writes a
  checkpoint, and exits 75.
- `resume: true` loads the newest checkpoint in `checkpoint_dir`, including
  optimizer and sampler state, so the run continues rather than restarting.
- `checkpoint_every` bounds how much work a sudden kill (no signal) can destroy.
  On interruptible instances keep it to a few minutes of steps.
- `keep_last_checkpoints` bounds disk use; checkpoints are written to a temporary
  file and renamed, so a partial write is never mistaken for a good checkpoint.

## Storage

Instance disks die with the instance. Keep packed shards and checkpoints in
object storage with cheap or free egress (Cloudflare R2, Backblaze B2), and sync
checkpoints on the same cadence you write them. Shards are immutable, so they
cache well; re-download rather than re-pack when you move machines.

## Cost hygiene

- Set `price_per_hour_usd` so every log line shows accumulated spend, and set
  `budget_usd` so the run stops instead of quietly overrunning.
- Prefer interruptible instances now that resume works; the discount is the whole
  point of having built checkpointing.
- Destroy the instance, not just the process. A stopped container with reserved
  storage can still bill.
- Record the resolved config and the final `tok/s` for every run so the next
  budget estimate starts from a measurement.
