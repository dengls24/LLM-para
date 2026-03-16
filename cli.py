"""
LLM-Para CLI — Command-line analysis interface
Usage: python cli.py --model "LLaMA-3 8B" --hardware "NVIDIA H100 SXM"
"""

import argparse
import json
import sys
from analyzer import LLMAnalyzer
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS


def format_number(v, unit=''):
    if v >= 1e18: return f"{v/1e18:.2f} E{unit}"
    if v >= 1e15: return f"{v/1e15:.2f} P{unit}"
    if v >= 1e12: return f"{v/1e12:.2f} T{unit}"
    if v >= 1e9:  return f"{v/1e9:.2f} G{unit}"
    if v >= 1e6:  return f"{v/1e6:.2f} M{unit}"
    if v >= 1e3:  return f"{v/1e3:.2f} K{unit}"
    return f"{v:.2f} {unit}"


def run_cli():
    parser = argparse.ArgumentParser(
        description='LLM-Para: Transformer FLOP & Roofline Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', '-m', type=str, help='Preset model name (use --list-models to see options)')
    parser.add_argument('--hardware', '-hw', type=str, default='NVIDIA H100 SXM', help='Hardware preset name')
    parser.add_argument('--seq-len', '-s', type=int, help='Override sequence length')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--quant', '-q', type=str, default='fp16', choices=['fp32','fp16','int8','int4','w4a8'],
                        help='Quantization preset')
    parser.add_argument('--flash-attn', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path')
    parser.add_argument('--json', type=str, help='Output JSON file path')
    parser.add_argument('--list-models', action='store_true', help='List available model presets')
    parser.add_argument('--list-hardware', action='store_true', help='List available hardware presets')
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable model presets:")
        for name in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[name]
            has_moe = '+ MoE' if cfg.get('num_experts_per_tok') else ''
            has_mla = '+ MLA' if cfg.get('use_mla') else ''
            print(f"  {name:<35} (L={cfg['num_layers']}, h={cfg['hidden_size']}, heads={cfg['num_heads']}{has_moe}{has_mla})")
        return

    if args.list_hardware:
        print("\nAvailable hardware presets:")
        for name, hw in HARDWARE_CONFIGS.items():
            peak = hw['peak_performance'] / 1e12
            bw = hw['memory_bandwidth'] / 1e9
            print(f"  {name:<40} Peak: {peak:.1f} TFLOP/s, BW: {bw:.0f} GB/s")
        return

    # Load model config
    if not args.model:
        print("Error: --model is required. Use --list-models to see options.")
        sys.exit(1)

    if args.model not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{args.model}'. Use --list-models to see options.")
        sys.exit(1)

    cfg = dict(MODEL_CONFIGS[args.model])

    # Apply overrides
    if args.seq_len:
        cfg['seq_len'] = args.seq_len
    cfg['batch_size'] = args.batch_size
    cfg['use_flash_attn'] = args.flash_attn

    # Quantization preset
    quant_presets = {
        'fp32': {'activation':32,'weight_attn':32,'weight_ffn':32,'kv_cache':32,'rope_bit':32},
        'fp16': {'activation':16,'weight_attn':16,'weight_ffn':16,'kv_cache':16,'rope_bit':32},
        'int8': {'activation': 8,'weight_attn': 8,'weight_ffn': 8,'kv_cache': 8,'rope_bit':32},
        'int4': {'activation': 8,'weight_attn': 4,'weight_ffn': 4,'kv_cache': 4,'rope_bit':32},
        'w4a8': {'activation': 8,'weight_attn': 4,'weight_ffn': 4,'kv_cache': 8,'rope_bit':32},
    }
    if args.quant in quant_presets:
        cfg['quant_config'] = quant_presets[args.quant]

    # Run analysis
    print(f"\n{'='*60}")
    print(f"  LLM-Para Analysis")
    print(f"  Model:    {args.model}")
    print(f"  Hardware: {args.hardware}")
    print(f"  Quant:    {args.quant}")
    print(f"{'='*60}")

    analyzer = LLMAnalyzer(cfg)
    results = analyzer.analyze()
    summary = analyzer.get_summary(results)

    # Print summary
    print(f"\n📊 Summary ({cfg['num_layers']}-layer, seq={cfg['seq_len']}, batch={cfg['batch_size']}):")
    print(f"  Total FLOPs:    {format_number(summary['total_flops'], 'FLOP')}")
    print(f"  Prefill FLOPs:  {format_number(summary['prefill_flops'], 'FLOP')}")
    print(f"  Decode FLOPs:   {format_number(summary['decode_flops'], 'FLOP')} (per token)")
    print(f"  Parameters:     {format_number(summary['total_params'], '')}")
    print(f"  Model Size:     {summary['model_size_gb']:.2f} GB")
    print(f"\n💾 KV Cache:")
    print(f"  Per token:      {summary['kv_bytes_per_token']:.0f} bytes")
    print(f"  Prefill:        {summary['kv_prefill_mb']:.1f} MB")
    print(f"  Max (decode):   {summary['kv_max_mb']:.1f} MB")
    print(f"  GQA ratio:      {summary['gqa_ratio']:.2f}× reduction")

    # Roofline analysis
    if args.hardware in HARDWARE_CONFIGS:
        hw = HARDWARE_CONFIGS[args.hardware]
        roofline = analyzer.get_roofline_data(results, hw)
        ridge = roofline['ridge_point']
        mem_ops = sum(1 for p in roofline['points'] if p['is_memory_bound'])
        cmp_ops = sum(1 for p in roofline['points'] if not p['is_memory_bound'])
        print(f"\n🎯 Roofline ({args.hardware}):")
        print(f"  Ridge point:    {ridge:.1f} FLOP/B")
        print(f"  Memory-bound:   {mem_ops} operators")
        print(f"  Compute-bound:  {cmp_ops} operators")

    # Top operators by FLOPs
    top_ops = sorted(results, key=lambda r: r['flops_total'], reverse=True)[:5]
    print(f"\n🔝 Top-5 Operators by FLOPs:")
    for i, op in enumerate(top_ops, 1):
        print(f"  {i}. [{op['phase']:7}] {op['operation']:<30} {format_number(op['flops_total'], 'FLOP'):>12}  density={op['density']:.1f}")

    # Export
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✅ CSV saved: {args.output}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'results': results, 'summary': summary}, f, indent=2)
        print(f"✅ JSON saved: {args.json}")

    print()


if __name__ == '__main__':
    run_cli()
