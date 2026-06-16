#!/usr/bin/env python3
"""30초 더미 음성 N회 → .blend 생성 벤치마크 + 모델 Parameters / GFLOPs."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import demo  # noqa: E402
from charsiu_ko.charsiu_predictive_aligner_ko import charsiu_predictive_aligner_ko  # noqa: E402


def format_elapsed(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds:.2f}s ({seconds / 60:.2f}min)"
    return f"{seconds:.2f}s"


def create_dummy_wav(path: str, duration_sec: float = 30.0, sr: int = 16000) -> str:
    samples = int(duration_sec * sr)
    audio = np.zeros(samples, dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sr)
    return path


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_wav2vec2_gflops(config, seq_len: int) -> float:
    """Profiler FLOPs 미지원 시 Wav2Vec2 transformer 구간 근사."""
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    intermediate = config.intermediate_size
    vocab = config.vocab_size

    # Self-attention (Q,K,V, out proj) + FFN, per layer
    attn_flops = 4 * seq_len * hidden * hidden + 2 * heads * seq_len * seq_len * (hidden // heads)
    ffn_flops = 2 * seq_len * hidden * intermediate
    per_layer = attn_flops + ffn_flops
    transformer_flops = layers * per_layer
    head_flops = seq_len * hidden * vocab
    return (transformer_flops + head_flops) / 1e9


def measure_gflops(model, processor, device, duration_sec: float, sr: int = 16000) -> tuple[float, int, str]:
    device = torch.device(device)
    samples = int(duration_sec * sr)
    audio = np.zeros(samples, dtype=np.float32)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_values"].shape[-1])

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            model(**inputs)

    total_flops = sum(
        evt.flops for evt in prof.key_averages() if evt.flops is not None and evt.flops > 0
    )

    if total_flops > 0:
        return total_flops / 1e9, input_len, "torch.profiler"

    with torch.no_grad():
        outputs = model.wav2vec2.feature_extractor(inputs["input_values"])
        seq_len = int(outputs.shape[-1])

    gflops = estimate_wav2vec2_gflops(model.config, seq_len)
    return gflops, input_len, f"estimate (seq_len={seq_len})"


def build_run_args(
    wav_path: str,
    run_idx: int,
    output_root: str,
    model_path: str,
    setting_dir: str,
    base_modelpath: str,
    comfort_viz_json: str,
    n_prob: int,
) -> argparse.Namespace:
    wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
    return argparse.Namespace(
        mode="prob",
        log=False,
        dir=True,
        n_prob=n_prob,
        input=wav_path,
        setting_dir=setting_dir,
        save_path=os.path.join(output_root, f"run_{run_idx:03d}"),
        base_modelpath=base_modelpath,
        model_path=model_path,
        gt=None,
        alpha_face=1.0,
        alpha_mouth=1.0,
        alpha_tongue=1.0,
        blend_tag=None,
        comfort_viz=True,
        comfort_viz_json=comfort_viz_json,
        comfort_alpha_front=0.7,
        comfort_alpha_side=0.4,
        comfort_strict=False,
        render=False,
        render_blender_bin=os.environ.get("BLENDER_BIN"),
        render_engine="auto",
        render_video_bitrate=8000,
        render_audio_bitrate="192k",
        render_muxer="auto",
        render_overwrite=True,
        render_keep_video_only=False,
        render_verbose=False,
        render_strict=False,
        _wav_stem=wav_stem,
    )


def run_single_blend(args, charsiu, lip_param, tongue_param, vocal2model, logvocab) -> tuple[float, str]:
    wav_stem = args._wav_stem
    result_root = os.path.join(args.save_path, wav_stem)
    _, result_prob_path, _ = demo.ensure_result_dirs(result_root, args.n_prob)
    final_name = demo.build_final_name(wav_stem, args.blend_tag)
    blend_path = os.path.join(result_prob_path, f"{final_name}.blend")

    start = time.perf_counter()
    demo.run_prob(
        file_path=args.input,
        output_dir=result_prob_path,
        final_name=final_name,
        args=args,
        charsiu=charsiu,
        lip_param=lip_param,
        tongue_param=tongue_param,
        vocal2model=vocal2model,
        logvocab=logvocab,
    )
    elapsed = time.perf_counter() - start

    if not os.path.isfile(blend_path):
        raise FileNotFoundError(f".blend not created: {blend_path}")

    return elapsed, blend_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark blend generation (30s dummy audio x N)")
    parser.add_argument("--duration", type=float, default=30.0, help="Dummy audio length (seconds)")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (excluded from total)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_DIR, "weights", "model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1"),
    )
    parser.add_argument("--setting_dir", default=os.path.join(PROJECT_DIR, "charsiu_ko"))
    parser.add_argument("--base_modelpath", default=os.path.join(PROJECT_DIR, "assets", "mouth.blend"))
    parser.add_argument("--comfort_viz_json", default=os.path.join(PROJECT_DIR, "vertex_groups_v3.json"))
    parser.add_argument("--output_root", default=os.path.join(PROJECT_DIR, "results", "benchmark_blend"))
    parser.add_argument("--dummy_wav", default=os.path.join(PROJECT_DIR, "benchmark_dummy_30s.wav"))
    parser.add_argument("--n_prob", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_path = os.path.abspath(args.model_path)

    weight_file = os.path.join(args.model_path, "model.safetensors")
    if not os.path.isfile(weight_file):
        alt = os.path.join(args.model_path, "pytorch_model.bin")
        if not os.path.isfile(alt):
            print(f"[ERROR] Model weights not found under {args.model_path}")
            print("        Expected model.safetensors or pytorch_model.bin")
            return 1

    print("=" * 60)
    print("Blend Generation Benchmark")
    print("=" * 60)

    create_dummy_wav(args.dummy_wav, duration_sec=args.duration, sr=args.sr)
    print(f"[INPUT] Dummy WAV: {args.dummy_wav} ({args.duration}s, {args.sr}Hz)")

    processor_path = os.path.join(args.model_path, "processor")
    vocab_path = os.path.join(args.model_path, "vocab.json")

    print("[MODEL] Loading acoustic model...")
    load_start = time.perf_counter()
    charsiu = charsiu_predictive_aligner_ko(
        model_path=args.model_path,
        processor_path=processor_path,
        vocab_path=vocab_path,
    )
    load_seconds = time.perf_counter() - load_start
    print(f"[MODEL] Loaded in {format_elapsed(load_seconds)}")

    total_params, trainable_params = count_parameters(charsiu.model)
    print(f"[MODEL] Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    gflops, input_len, gflops_method = measure_gflops(
        charsiu.model,
        charsiu.processor,
        charsiu.device,
        duration_sec=args.duration,
        sr=args.sr,
    )
    print(f"[MODEL] GFLOPs (1 forward, {args.duration}s / {input_len} samples): {gflops:.3f} ({gflops_method})")

    lip_param, tongue_param, vocal2model, vocab = demo.INIT_SETTING(args.setting_dir)
    logvocab = [None] * len(vocab)
    for phone, idx in vocab.items():
        if idx < len(logvocab):
            logvocab[idx] = phone

    os.makedirs(args.output_root, exist_ok=True)

    all_runs = args.warmup + args.runs
    run_times: list[float] = []

    print("-" * 60)
    print(f"[BENCH] Warmup={args.warmup}, measured runs={args.runs}")
    for i in range(1, all_runs + 1):
        run_args = build_run_args(
            wav_path=args.dummy_wav,
            run_idx=i,
            output_root=args.output_root,
            model_path=args.model_path,
            setting_dir=args.setting_dir,
            base_modelpath=args.base_modelpath,
            comfort_viz_json=args.comfort_viz_json,
            n_prob=args.n_prob,
        )
        elapsed, blend_path = run_single_blend(
            run_args, charsiu, lip_param, tongue_param, vocal2model, logvocab
        )
        label = "warmup" if i <= args.warmup else f"run {i - args.warmup:02d}"
        print(f"  [{label}] {format_elapsed(elapsed)} -> {blend_path}")
        if i > args.warmup:
            run_times.append(elapsed)

    total_seconds = sum(run_times)
    avg_seconds = total_seconds / len(run_times) if run_times else 0.0

    print("=" * 60)
    print("[RESULT] Blend generation benchmark (model load excluded)")
    print(f"  Input       : {args.duration}s dummy audio x {args.runs} runs")
    print(f"  Total time  : {format_elapsed(total_seconds)}")
    print(f"  Avg / run   : {format_elapsed(avg_seconds)}")
    print(f"  Min / Max   : {format_elapsed(min(run_times))} / {format_elapsed(max(run_times))}")
    print("[RESULT] Acoustic model (Wav2Vec2ForCTC)")
    print(f"  Parameters  : {total_params:,}")
    print(f"  GFLOPs      : {gflops:.3f} per forward ({gflops_method})")
    print(f"  Model load  : {format_elapsed(load_seconds)} (not included in total)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
