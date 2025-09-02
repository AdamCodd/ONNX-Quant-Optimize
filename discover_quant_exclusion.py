# discover_quant_exclusions.py

"""
A two-stage workflow to discover the optimal set of nodes to exclude from quantization.
Supports searching for sensitive nodes in the encoder, decoder, or both.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType
from transformers import AutoTokenizer, AutoConfig

# Local imports
from quant_utils import (
    SuppressQuantFilter, find_nodes_recursively, ensure_named_nodes,
    load_samples_from_jsonl, get_filesize_mb
)
from quant_engine import (
    run_benchmark, dynamic_quantization, find_optimal_exclusions, apply_quantization_from_exclusions_file,
    HIGHER_IS_BETTER_METRICS
)

if __name__ == "__main__":
    script_start_time = time.perf_counter()
    
    # Setup logging
    root_logger = logging.getLogger()
    root_logger.addFilter(SuppressQuantFilter())
    
    # ==== DEFAULTS (overridden by config file) ====
    DEFAULTS = {
        "candidate_op_types": [
            "Gather", "Gemm", "MatMul", "Add", "Sub", "Mul", "Softmax", "LayerNormalization",
            "Gelu", "Div", "Exp", "Pow", "Sqrt", "ReduceMean", "Slice", "Unsqueeze",
            "Transpose", "Concat", "Reshape", "Cast"
        ],
        "enable_subgraph": True,
        "execution_provider": "CPUExecutionProvider",
        "fp32_decoder": "decoder_model_merged.onnx",
        "fp32_encoder": "encoder_model.onnx",
        "max_generation_length": 100,
        "max_nodes_to_exclude": None,
        "metrics": ["wer", "cer"],
        "model_dir": ".",
        "model_reference": "safetensors",
        "multiprocessing": True,
        "onnx_dir": "onnx-model",
        "primary_metric": "wer",
        "quant_test_dir": "onnx-quant-discovery",
        "quant_type": "QUInt8",
        "resume": False,
        "samples_jsonl": "samples.jsonl",
        "search_target": "both",
        "strategy_stage1": "first",
        "strategy_stage2": "relaxed",
        "target": None,
        "task": "text2text-generation",
        "with-past": True,
        "workers": None # None means autodetect
    }

    script_start_time = time.perf_counter()
    # Get the script's directory to resolve default paths
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Discover nodes to exclude from quantization for encoder-decoder ONNX models.")
    parser.add_argument("--config", "-c", type=str, default=str(script_dir / "config_quant.json"), help="Path to JSON config file (overrides defaults).")
    parser.add_argument("--results", type=str, default=str(script_dir / "results.jsonl"), help="Path to export the final exclusion lists as a JSONL file.")
    parser.add_argument("--export_final", type=str, help="Path to a directory to export the final quantized ONNX models with the discovered exclusions.")
    parser.add_argument("--export_excluded", type=str, help="Path to a results.jsonl file. Skips the search and applies the exclusions from this file to quantize the model.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

    # Add file handler to log everything to a file in the script directory
    log_file_path = script_dir / "quant-optimizer.log"
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)

    # Silence the overly verbose onnxruntime logger
    ort_logger = logging.getLogger("onnxruntime")
    ort_logger.setLevel(logging.ERROR)

    cfg_path = Path(args.config)
    cfg = DEFAULTS.copy()
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh: cfg.update(json.load(fh))
    else:
        logging.warning(f"Config file {cfg_path} not found. Using defaults.")

    # --- Load configuration ---
    # Ensure all relative paths in the config are resolved relative to the script's location
    MODEL_DIR = Path(cfg["model_dir"])
    if not MODEL_DIR.is_absolute():
        MODEL_DIR = script_dir / MODEL_DIR

    ONNX_DIR = Path(cfg["onnx_dir"])
    if not ONNX_DIR.is_absolute():
        ONNX_DIR = script_dir / ONNX_DIR
    
    QUANT_TEST_DIR = Path(cfg["quant_test_dir"])
    if not QUANT_TEST_DIR.is_absolute():
        QUANT_TEST_DIR = script_dir / QUANT_TEST_DIR

    samples_jsonl = Path(cfg["samples_jsonl"])
    if not samples_jsonl.is_absolute():
        samples_jsonl = script_dir / samples_jsonl
        
    PRIMARY_METRIC, METRICS = cfg["primary_metric"], cfg["metrics"]
    EXECUTION_PROVIDER, MAX_GENERATION_LENGTH = cfg["execution_provider"], cfg["max_generation_length"]
    QUANT_TYPE_STR, SEARCH_TARGET = cfg["quant_type"], cfg["search_target"]
    TARGET = cfg["target"]
    DEVICE = "cpu" if EXECUTION_PROVIDER == "CPUExecutionProvider" else "cuda"

    if cfg["model_reference"] in ("safetensors", "pytorch", "pt"):
        if not (ONNX_DIR / cfg["fp32_encoder"]).exists() or not (ONNX_DIR / cfg["fp32_decoder"]).exists():
            logging.info(f"Exporting PyTorch/safetensors model from '{MODEL_DIR}' to ONNX in '{ONNX_DIR}'...")
            try:
                from optimum.exporters.onnx import main_export
                export_task = cfg["task"] + ("-with-past" if cfg.get("with-past", True) else "")
                main_export(
                    model_name_or_path=str(MODEL_DIR),
                    output=str(ONNX_DIR),
                    task=export_task,
                    framework="pt",
                    device=DEVICE,
                    monolith=False,
                    no_post_process=False,
                )

                logging.info("✅ Export complete.")
            except ImportError:
                logging.error("The 'optimum[exporters]' library is required. Please run: pip install optimum[exporters]")
                sys.exit(1)
        else:
            logging.info(f"ONNX models found in '{ONNX_DIR}'. Skipping export.")
    elif cfg["model_reference"] == "onnx-fp32":
        logging.info("Model reference set to 'onnx-fp32' — expecting ONNX FP32 models to already exist in the ONNX_DIR.")
    else:
        logging.warning(f"Unknown model_reference '{cfg['model_reference']}'. If your reference is a PyTorch model, set 'model_reference' to 'pytorch' or 'safetensors' in the config.")

    fp32_encoder_path, fp32_decoder_path = ONNX_DIR / cfg["fp32_encoder"], ONNX_DIR / cfg["fp32_decoder"]
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    QUANT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    if not fp32_encoder_path.exists() or not fp32_decoder_path.exists():
        raise FileNotFoundError(
            f"Missing FP32 ONNX model. Check paths in config: {fp32_encoder_path}, {fp32_decoder_path} "
            f"or set 'model_reference' to 'pytorch'|'safetensors' (to export from MODEL_DIR) or 'onnx-fp32' (if ONNX already present)."
        )

    # --- Pre-load assets for either workflow ---
    if QUANT_TYPE_STR == "QUInt8":
        quant_type, quant_suffix = QuantType.QUInt8, "quint8"
    elif QUANT_TYPE_STR == "QInt8":
        quant_type, quant_suffix = QuantType.QInt8, "qint8"
    else:
        raise ValueError(f"Unsupported 'quant_type' in config: '{QUANT_TYPE_STR}'. Must be 'QUInt8' or 'QInt8'.")

    tokenizer, config = AutoTokenizer.from_pretrained(str(MODEL_DIR)), AutoConfig.from_pretrained(str(MODEL_DIR))
    samples = load_samples_from_jsonl(samples_jsonl)
    logging.info(f"Loaded {len(samples)} sample(s) from {samples_jsonl} for benchmarking.\n")
    
    run_benchmark_args = {
        "tokenizer": tokenizer, "config": config, "samples": samples,
        "metrics": METRICS, "execution_provider": EXECUTION_PROVIDER,
        "max_generation_length": MAX_GENERATION_LENGTH,
        "quant_test_dir": QUANT_TEST_DIR
    }

    # Handle direct quantization from exclusions file and run comparison
    if args.export_excluded:
        export_dir = script_dir / "onnx_excluded"
        if args.export_final:
            export_dir = Path(args.export_final)
        
        # Apply the quantization using the provided exclusion file
        apply_quantization_from_exclusions_file(
            exclusions_file=Path(args.export_excluded),
            export_dir=export_dir,
            onnx_dir=ONNX_DIR,
            cfg=cfg
        )
        
        # --- Run comparison benchmark ---
        logging.info("\n" + "="*20 + " BENCHMARK COMPARISON " + "="*20)
        
        # 1. Benchmark FP32 model
        logging.info("🔬 Benchmarking FP32 ONNX Model (Reference)...")
        reference_metrics, reference_time = run_benchmark(fp32_encoder_path, fp32_decoder_path, **run_benchmark_args)
        logging.info(f"✅ Reference (FP32) Metrics: {json.dumps({k: round(v, 4) for k, v in reference_metrics.items()})}")
        logging.info(f"✅ Inference Time: {reference_time:.4f}s")

        # 2. Benchmark newly quantized model
        logging.info(f"🔬 Benchmarking Quantized ONNX Model from {export_dir}...")
        quant_encoder_path = export_dir / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic_final.onnx")
        quant_decoder_path = export_dir / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic_final.onnx")
        
        if not quant_encoder_path.exists() or not quant_decoder_path.exists():
            logging.error(f"Quantized models not found in {export_dir}. Cannot run benchmark.")
        else:
            quant_metrics, quant_time = run_benchmark(quant_encoder_path, quant_decoder_path, **run_benchmark_args)
            logging.info(f"✅ Quantized Metrics: {json.dumps({k: round(v, 4) for k, v in quant_metrics.items()})}")
            logging.info(f"✅ Inference Time: {quant_time:.4f}s")
            logging.info("="*62)

        script_end_time = time.perf_counter()
        logging.info(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds.")
        sys.exit(0)

    # --- The rest of the script remains unchanged for the search workflow ---
    logging.info(f"Search Target: '{SEARCH_TARGET}'")
    logging.info(f"Using Stage 1 strategy: '{cfg['strategy_stage1']}'")
    if cfg['strategy_stage1'] == "percent":
        if TARGET is None or not (0.0 <= TARGET <= 1.0):
            raise ValueError("For 'percent' strategy, 'target' must be set in the config file as a float between 0.0 and 1.0.")

    expanded_primary_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'] if 'rouge' in METRICS else []
    expanded_primary_metrics.extend([m for m in METRICS if m != 'rouge'])
    if PRIMARY_METRIC not in expanded_primary_metrics:
        raise ValueError(f"Primary metric '{PRIMARY_METRIC}' must be one of the tested metrics: {expanded_primary_metrics}")

    minimize_metric = PRIMARY_METRIC not in HIGHER_IS_BETTER_METRICS
    quant_extra_options = {"EnableSubgraph": cfg["enable_subgraph"]}
    
    logging.info("🔬 Benchmarking FP32 ONNX Model (Reference)...")
    reference_metrics, reference_time = run_benchmark(fp32_encoder_path, fp32_decoder_path, **run_benchmark_args)
    REFERENCE_SCORE = reference_metrics[PRIMARY_METRIC]
    fp32_filesize = get_filesize_mb(fp32_encoder_path, fp32_decoder_path)
    logging.info(f"✅ Reference (FP32) Metrics: {json.dumps({k: round(v, 4) for k, v in reference_metrics.items()})}")
    logging.info(f"✅ Inference Time: {reference_time:.4f}s, Size: {fp32_filesize:.2f}MB")


    logging.info("🔬 Benchmarking Fully Quantized Model (Baseline)...")
    baseline_quant_encoder_path = QUANT_TEST_DIR / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic.onnx")
    baseline_quant_decoder_path = QUANT_TEST_DIR / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic.onnx")
    dynamic_quantization(fp32_encoder_path, baseline_quant_encoder_path, quant_type, [], quant_extra_options)
    dynamic_quantization(fp32_decoder_path, baseline_quant_decoder_path, quant_type, [], quant_extra_options)
    baseline_metrics, baseline_time = run_benchmark(baseline_quant_encoder_path, baseline_quant_decoder_path, **run_benchmark_args)
    BASELINE_SCORE = baseline_metrics[PRIMARY_METRIC]
    baseline_quant_filesize = get_filesize_mb(baseline_quant_encoder_path, baseline_quant_decoder_path)
    logging.info(f"✅ Baseline (Fully Quantized) Metrics: {json.dumps({k: round(v, 4) for k, v in baseline_metrics.items()})}")
    logging.info(f"✅ Inference Time: {baseline_time:.4f}s, Size: {baseline_quant_filesize:.2f}MB\n")


    decoder_exclusion_list, encoder_exclusion_list = [], []
    encoder_final_score, decoder_final_score = None, None
    
    search_args = {
        "reference_score": REFERENCE_SCORE, "run_benchmark_args": run_benchmark_args, "cfg": cfg,
        "quant_type": quant_type, "quant_extra_options": quant_extra_options,
        "minimize_metric": minimize_metric, "primary_metric": PRIMARY_METRIC,
        "quant_test_dir": QUANT_TEST_DIR
    }

    # When search_target is 'both', we check the encoder first, then the decoder.
    if SEARCH_TARGET in ["encoder", "both"]:
        logging.info("\n" + ">>>--- STARTING SEARCH FOR ENCODER ---<<<")
        logging.info("🔬 Establishing baseline for encoder search (quantized encoder + FP32 decoder)...")
        encoder_search_baseline_metrics, _ = run_benchmark(baseline_quant_encoder_path, fp32_decoder_path, **run_benchmark_args)
        encoder_search_baseline_score = encoder_search_baseline_metrics[PRIMARY_METRIC]
        logging.info(f"✅ Encoder Search Baseline {PRIMARY_METRIC.upper()}: {encoder_search_baseline_score:.4f}")

        # Check if the baseline already meets the reference performance
        fp32_level_met = (encoder_search_baseline_score <= REFERENCE_SCORE) if minimize_metric else (encoder_search_baseline_score >= REFERENCE_SCORE)

        if fp32_level_met:
            logging.info(f"✅ Encoder search baseline ({encoder_search_baseline_score:.4f}) already meets/exceeds the FP32 reference ({REFERENCE_SCORE:.4f}).")
            logging.info("Skipping node exclusion search for the encoder as it's not needed.")
            encoder_exclusion_list = []
        else:
            logging.info("Loading encoder model once to extract node names...")
            encoder_model_onnx = onnx.load(str(fp32_encoder_path))

            if not ensure_named_nodes(encoder_model_onnx):
                logging.error("Encoder ONNX nodes appear to lack names. Node-based exclusion search will be skipped for the encoder. Consider re-exporting with node names.")
                encoder_exclusion_list = []
            else:
                encoder_nodes_by_type = {op: [n.name for n in find_nodes_recursively(encoder_model_onnx.graph, [op]) if n.name] for op in cfg["candidate_op_types"]}

                encoder_exclusion_list, encoder_final_score = find_optimal_exclusions(
                    model_to_search_path=fp32_encoder_path,
                    nodes_by_type=encoder_nodes_by_type,
                    part_name="encoder",
                    fixed_encoder_path=None,  # Not used when searching encoder
                    fixed_decoder_path=fp32_decoder_path,  # Use FP32 decoder to isolate encoder issues
                    baseline_score=encoder_search_baseline_score,
                    **search_args
                )

    if SEARCH_TARGET in ["decoder", "both"]:
        logging.info("\n" + ">>>--- STARTING SEARCH FOR DECODER ---<<<")

        logging.info("🔬 Establishing baseline for decoder search (FP32 encoder + quantized decoder)...")
        decoder_search_baseline_metrics, _ = run_benchmark(fp32_encoder_path, baseline_quant_decoder_path, **run_benchmark_args)
        decoder_search_baseline_score = decoder_search_baseline_metrics[PRIMARY_METRIC]
        logging.info(f"✅ Decoder Search Baseline {PRIMARY_METRIC.upper()}: {decoder_search_baseline_score:.4f}")

        # Check if the baseline already meets the reference performance
        fp32_level_met = (decoder_search_baseline_score <= REFERENCE_SCORE) if minimize_metric else (decoder_search_baseline_score >= REFERENCE_SCORE)

        if fp32_level_met:
            logging.info(f"✅ Decoder search baseline ({decoder_search_baseline_score:.4f}) already meets/exceeds the FP32 reference ({REFERENCE_SCORE:.4f}).")
            logging.info("Skipping node exclusion search for the decoder as it's not needed.")
            decoder_exclusion_list = []
        else:
            logging.info("Loading decoder model once to extract node names...")
            decoder_model_onnx = onnx.load(str(fp32_decoder_path))

            if not ensure_named_nodes(decoder_model_onnx):
                logging.error("Decoder ONNX nodes appear to lack names. Node-based exclusion search will be skipped for the decoder. Consider re-exporting with node names.")
                decoder_exclusion_list = []
            else:
                decoder_nodes_by_type = {op: [n.name for n in find_nodes_recursively(decoder_model_onnx.graph, [op]) if n.name] for op in cfg["candidate_op_types"]}

                decoder_exclusion_list, decoder_final_score = find_optimal_exclusions(
                    model_to_search_path=fp32_decoder_path,
                    nodes_by_type=decoder_nodes_by_type,
                    part_name="decoder",
                    fixed_encoder_path=fp32_encoder_path,  # Use FP32 encoder to isolate decoder issues
                    fixed_decoder_path=None,  # Not used when searching decoder
                    baseline_score=decoder_search_baseline_score,
                    **search_args
                )

    if SEARCH_TARGET == "both" and encoder_exclusion_list and decoder_exclusion_list:
        logging.info("\n>>>--- COMPARING ENCODER-ONLY VS DECODER-ONLY OPTIMIZATION ---<<<")
        logging.info(f"Encoder-only optimization score ({PRIMARY_METRIC.upper()}): {encoder_final_score:.4f}")
        logging.info(f"Decoder-only optimization score ({PRIMARY_METRIC.upper()}): {decoder_final_score:.4f}")

        # Decide which exclusion list to keep based on which one gave a better result
        # in its isolated test.
        if minimize_metric:
            # Lower is better
            if decoder_final_score < encoder_final_score:
                logging.info("Decoder-only optimization is better. Discarding encoder exclusions.")
                encoder_exclusion_list = []
            else:
                logging.info("Encoder-only optimization is better or equal. Discarding decoder exclusions.")
                decoder_exclusion_list = []
        else:
            # Higher is better
            if decoder_final_score > encoder_final_score:
                logging.info("Decoder-only optimization is better. Discarding encoder exclusions.")
                encoder_exclusion_list = []
            else:
                logging.info("Encoder-only optimization is better or equal. Discarding decoder exclusions.")
                decoder_exclusion_list = []


    # --- Run a final benchmark with the optimal exclusion lists ---
    logging.info("\n🔬 Benchmarking final configuration with optimal exclusions...")
    final_quant_encoder_path = QUANT_TEST_DIR / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic_final_benchmark.onnx")
    final_quant_decoder_path = QUANT_TEST_DIR / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic_final_benchmark.onnx")
    final_metrics, final_time, final_quant_filesize = {}, 0.0, 0.0
    try:
        # Quantize encoder with its final exclusion list
        dynamic_quantization(fp32_encoder_path, final_quant_encoder_path, quant_type, encoder_exclusion_list, quant_extra_options)
        # Quantize decoder with its final exclusion list
        dynamic_quantization(fp32_decoder_path, final_quant_decoder_path, quant_type, decoder_exclusion_list, quant_extra_options)
        # Run the benchmark on the final configuration
        final_metrics, final_time = run_benchmark(final_quant_encoder_path, final_quant_decoder_path, **run_benchmark_args)
        final_quant_filesize = get_filesize_mb(final_quant_encoder_path, final_quant_decoder_path)
        delta_baseline = final_metrics[PRIMARY_METRIC] - REFERENCE_SCORE
        logging.info(f"✅ Final Optimized Metrics: {json.dumps({k: round(v, 4) for k, v in final_metrics.items()})}")
        logging.info(f"✅ Inference Time: {final_time:.4f}s, Size: {final_quant_filesize:.2f}MB")
        logging.info(f"✅ Delta from FP32 Baseline ({PRIMARY_METRIC.upper()}): {delta_baseline:+.4f}")

    except Exception as e:
        logging.error(f"Failed to run final benchmark: {e}. Final metrics will be empty.")
        delta_baseline = 0.0
    finally:
        # Clean up the temporary benchmark files
        if final_quant_encoder_path.exists(): final_quant_encoder_path.unlink()
        if final_quant_decoder_path.exists(): final_quant_decoder_path.unlink()

    # --- Final Recommendations ---
    logging.info("\n\n" + "="*25 + " FINAL RESULTS " + "="*25)
    if decoder_exclusion_list:
        logging.info(f"Found minimal exclusion list for DECODER with {len(decoder_exclusion_list)} nodes.")
        logging.info("RECOMMENDATION: Use this list for the 'nodes_to_exclude' argument when quantizing the decoder.")
        print("-" * 80)
        print("DECODER_NODES_TO_EXCLUDE = [")
        for node_name in sorted(decoder_exclusion_list): print(f"    '{node_name}',")
        print("]")
        print("-" * 80)
    elif SEARCH_TARGET in ["decoder", "both"]:
        logging.info("No beneficial nodes to exclude were found for the DECODER.")

    if encoder_exclusion_list:
        logging.info(f"\nFound minimal exclusion list for ENCODER with {len(encoder_exclusion_list)} nodes.")
        logging.info("RECOMMENDATION: Use this list for the 'nodes_to_exclude' argument when quantizing the encoder.")
        print("-" * 80)
        print("ENCODER_NODES_TO_EXCLUDE = [")
        for node_name in sorted(encoder_exclusion_list): print(f"    '{node_name}',")
        print("]")
        print("-" * 80)
    elif SEARCH_TARGET in ["encoder", "both"]:
        logging.info("\nNo beneficial nodes to exclude were found for the ENCODER.")

    # --- Final Export Actions ---
    # 1. Export results.jsonl
    results_path = Path(args.results)
    
    results_data = {
        "metrics": METRICS,
        "baseline_fp32": {
            "metrics": {k: round(v, 4) for k, v in reference_metrics.items()},
            "time_seconds": round(reference_time, 4),
            "filesize_mb": round(fp32_filesize, 2)
        },
        f"baseline_{quant_suffix}": {
            "metrics": {k: round(v, 4) for k, v in baseline_metrics.items()},
            "time_seconds": round(baseline_time, 4),
            "filesize_mb": round(baseline_quant_filesize, 2)
        },
        "encoder_nodes_to_exclude": sorted(encoder_exclusion_list),
        "decoder_nodes_to_exclude": sorted(decoder_exclusion_list),
        f"final_results_{quant_suffix}": {
            "metrics": {k: round(v, 4) for k, v in final_metrics.items()} if final_metrics else {},
            "time_seconds": round(final_time, 4),
            "filesize_mb": round(final_quant_filesize, 2)
        },
        "delta_baseline": round(delta_baseline, 4)
    }
    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(results_data) + "\n")
        logging.info(f"\n✅ Successfully exported results to {results_path}")
    except Exception as e:
        logging.error(f"\n❌ Failed to export results to {results_path}: {e}")

    # 2. Export final ONNX models if requested
    if args.export_final:
        export_dir = Path(args.export_final)
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"\nExporting final quantized models to {export_dir}...")

            final_quant_encoder_path = export_dir / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic_final.onnx")
            final_quant_decoder_path = export_dir / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic_final.onnx")

            logging.info(f"Quantizing encoder with {len(encoder_exclusion_list)} excluded nodes...")
            dynamic_quantization(fp32_encoder_path, final_quant_encoder_path, quant_type, encoder_exclusion_list, quant_extra_options)
            logging.info(f"  -> Saved to {final_quant_encoder_path}")

            logging.info(f"Quantizing decoder with {len(decoder_exclusion_list)} excluded nodes...")
            dynamic_quantization(fp32_decoder_path, final_quant_decoder_path, quant_type, decoder_exclusion_list, quant_extra_options)
            logging.info(f"  -> Saved to {final_quant_decoder_path}")
            logging.info("✅ Final model export complete.")

        except Exception as e:
            logging.error(f"\n❌ Failed to export final ONNX models to {export_dir}: {e}")

    # --- Final Summary Table ---
    logging.info("\n\n" + "📊" + "="*14 + " FINAL SUMMARY " + "="*14 + "📊")
    display_metrics = sorted(reference_metrics.keys())
    
    # Header
    header = f"{'Model':<40} {'Time (s)':>10} {'ΔTime':>20}"
    for m in display_metrics: header += f" {m.upper():>9}"
    for m in display_metrics: header += f" {'Δ' + m.upper():>9}"
    header += f" {'Filesize (total)':>10}"
    logging.info(header)
    logging.info("-" * len(header))

    # FP32 Row
    fp32_row = f"{'FP32 Reference':<40} {reference_time:>10.4f} {'-':>20}"
    for m in display_metrics: fp32_row += f" {reference_metrics.get(m, 0):>9.4f}"
    for _ in display_metrics: fp32_row += f" {'-':>9}"
    fp32_row += f" {fp32_filesize:>9.2f}MB"
    logging.info(fp32_row)

    # Fully Quantized Row
    time_delta = baseline_time - reference_time
    speed_factor = reference_time / baseline_time if baseline_time > 0 else 0
    time_delta_str = f"{time_delta:+.3f}s (x{speed_factor:.2f})"
    
    quant_row = f"{QUANT_TYPE_STR.upper() + ' Dynamic (Fully Quantized)':<40} {baseline_time:>10.4f} {time_delta_str:>20}"
    for m in display_metrics: quant_row += f" {baseline_metrics.get(m, 0):>9.4f}"
    for m in display_metrics:
        metric_delta = baseline_metrics.get(m, 0) - reference_metrics.get(m, 0)
        quant_row += f" {metric_delta:>+9.4f}"
    quant_row += f" {baseline_quant_filesize:>9.2f}MB"
    logging.info(quant_row)

    # Final Partially Quantized Row
    if final_metrics:
        time_delta = final_time - reference_time
        speed_factor = reference_time / final_time if final_time > 0 else 0
        time_delta_str = f"{time_delta:+.3f}s (x{speed_factor:.2f})"
        
        final_row = f"{QUANT_TYPE_STR.upper() + ' Dynamic (Partially Quantized)':<40} {final_time:>10.4f} {time_delta_str:>20}"
        for m in display_metrics: final_row += f" {final_metrics.get(m, 0):>9.4f}"
        for m in display_metrics:
            metric_delta = final_metrics.get(m, 0) - reference_metrics.get(m, 0)
            final_row += f" {metric_delta:>+9.4f}"
        final_row += f" {final_quant_filesize:>9.2f}MB"
        logging.info(final_row)

    if not decoder_exclusion_list and not encoder_exclusion_list:
        logging.warning("\nThe script finished, but did not find any nodes to exclude that improved performance.")

    script_end_time = time.perf_counter()
    total_time = script_end_time - script_start_time
    logging.info(f"\nTotal script execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")
