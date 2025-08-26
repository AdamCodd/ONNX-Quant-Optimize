#!/usr/bin/env python3
"""
A two-stage workflow to discover the optimal set of nodes to exclude from quantization.
Supports searching for sensitive nodes in the encoder, decoder, or both.
"""
import argparse
import json
import logging
import sys
import tempfile
import os
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Any

import onnx
import onnxruntime as ort

class SuppressQuantFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if 'Quantization parameters for tensor' in msg:
            return False
        if 'Ignore MatMul due to non constant B' in msg:
            return False
        if 'Please consider to run pre-processing before quantization.' in msg:
            return False
        return True

root_logger = logging.getLogger()
root_logger.addFilter(SuppressQuantFilter())

from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# ==== CONSTANTS ====
HIGHER_IS_BETTER_METRICS = {"accuracy", "bleu", "rouge", "rouge1", "rouge2", "rougeL", "rougeLsum"}

# ==== DEFAULTS (overridden by config file) ====
DEFAULTS = {
    "model_reference": "safetensors",  # "safetensors" or "onnx-fp32"
    "model_dir": ".",
    "onnx_dir": "onnx-model",
    "quant_test_dir": "onnx-quant-discovery",
    "fp32_encoder": "encoder_model.onnx",
    "fp32_decoder": "decoder_model_merged.onnx",
    "execution_provider": "CPUExecutionProvider",
    "quant_type": "QUInt8",
    "search_target": "both", # "decoder", "encoder", or "both"
    "max_generation_length": 100,
    "samples_jsonl": "samples.jsonl",
    "task": "text2text-generation",
    "with-past": True,
    "enable_subgraph": True,
    "metrics": ["wer", "cer"],
    "primary_metric": "wer",
    "candidate_op_types": [
        "Gather", "Gemm", "MatMul", "Add", "Sub", "Mul", "Softmax", "LayerNormalization",
        "Gelu", "Div", "Exp", "Pow", "Sqrt", "ReduceMean", "Slice", "Unsqueeze",
        "Transpose", "Concat", "Reshape", "Cast"
    ],
    "target": None,
    "strategy_stage1": "first", # 'first', 'best', 'percent'
    "strategy_stage2": "relaxed", # 'relaxed', 'strict'
    "max_nodes_to_exclude": None # Integer limit or None
}


def find_nodes_recursively(graph: onnx.GraphProto, op_types: List[str]) -> List[onnx.NodeProto]:
    """Recursively find all nodes of specified types in a graph and its subgraphs."""
    found_nodes = []
    for node in graph.node:
        if node.op_type in op_types:
            found_nodes.append(node)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                found_nodes.extend(find_nodes_recursively(attr.g, op_types))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    found_nodes.extend(find_nodes_recursively(subgraph, op_types))
    return found_nodes


def _extract_sample_fields(sample: Dict) -> Tuple[str, str]:
    """Get (input_prompt, ground_truth) from a sample dict using flexible key names."""
    input_keys = ("input", "input_prompt", "prompt")
    gt_keys = ("ground_truth", "reference")
    input_prompt = None
    ground_truth = None
    for k in input_keys:
        if k in sample:
            input_prompt = sample[k]
            break
    for k in gt_keys:
        if k in sample:
            ground_truth = sample[k]
            break
    if input_prompt is None or ground_truth is None:
        raise ValueError("Each sample in the jsonl file must contain both an input and a ground truth. "
                         "Accepted input keys: 'input'|'input_prompt'|'prompt'. "
                         "Accepted ground truth keys: 'ground_truth'|'reference'.")
    return input_prompt, ground_truth


def run_benchmark(
    encoder_path: Path,
    decoder_path: Path,
    tokenizer: AutoTokenizer,
    config: AutoConfig,
    samples: List[Dict],
    metrics: List[str],
    execution_provider: str,
    max_generation_length: int
) -> Tuple[Dict[str, float], float]:
    """
    Runs inference on an encoder-decoder ONNX model pair and returns a dictionary of averaged metrics and the inference time.
    Inference is performed in a single batch for all samples to improve performance.
    """
    # Conditionally import evaluation libraries based on requested metrics
    load_metric = None
    if any(m in metrics for m in ["bleu", "rouge"]):
        try:
            from evaluate import load as load_metric
        except ImportError:
            logging.warning("Warning: 'evaluate' library not found. Please run 'pip install evaluate'. Metrics like 'bleu' and 'rouge' will not be available.")

    wer, cer = None, None
    if any(m in metrics for m in ["wer", "cer"]):
        try:
            from jiwer import wer, cer
        except ImportError:
            logging.warning("Warning: 'jiwer' library not found. Please run 'pip install jiwer'. Metrics 'wer' and 'cer' will not be available.")

    expanded_metrics = []
    for m in metrics:
        if m == 'rouge':
            expanded_metrics.extend(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
        else:
            expanded_metrics.append(m)
    scores = {metric: [] for metric in expanded_metrics}
    
    metric_calculators = {}
    if load_metric:
        for metric in metrics:
            if metric not in ["wer", "cer", "accuracy"]:
                try:
                    metric_calculators[metric] = load_metric(metric)
                except Exception as e:
                    logging.warning(f"Could not load metric '{metric}': {e}. Skipping.")

    start_time = time.perf_counter()
    try:
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 3
        encoder_sess = ort.InferenceSession(str(encoder_path), providers=[execution_provider], sess_options=sess_opt)
        decoder_sess = ort.InferenceSession(str(decoder_path), providers=[execution_provider], sess_options=sess_opt)
    except Exception as e:
        logging.error(f"Failed to create ONNX Runtime sessions: {e}")
        return {metric: 1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0 for metric in expanded_metrics}, 0.0

    try:
        # ---- BATCH PREPARATION ----
        input_prompts, ground_truths = zip(*[_extract_sample_fields(s) for s in samples])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(list(input_prompts), return_tensors="np", padding=True, truncation=True)
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        batch_size = input_ids.shape[0]

        encoder_hidden_states = encoder_sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        
        # ---- DECODER SETUP ----
        decoder_input_details = decoder_sess.get_inputs()
        encoder_seq_len = encoder_hidden_states.shape[1]
        past_decoder_seq_len = 1

        past_kv = {}
        for inp in decoder_input_details:
            name = inp.name
            if 'past_key_values' not in name: continue
            raw_shape = list(inp.shape)
            shape = []
            for idx, dim in enumerate(raw_shape):
                if isinstance(dim, int): shape.append(dim); continue
                if idx == 0: shape.append(batch_size)
                elif '.encoder.' in name: shape.append(encoder_seq_len)
                elif '.decoder.' in name: shape.append(past_decoder_seq_len)
                else: shape.append(1)
            dtype = np.float32
            past_kv[name] = np.zeros(tuple(shape), dtype=dtype)

        encoder_past_saved = {k: v for k, v in past_kv.items() if '.encoder.' in k}
        decoder_past_saved = {k: v for k, v in past_kv.items() if '.decoder.' in k}

        present_to_past = {}
        input_names = {inp.name for inp in decoder_input_details}
        output_infos = decoder_sess.get_outputs()
        for out in output_infos:
            oname = out.name
            if 'present' not in oname: continue
            candidates = [ oname.replace(p, 'past_key_values') for p in ['present', 'present.', '.present.', 'present_'] ]
            for c in candidates:
                if c in input_names:
                    present_to_past[oname] = c
                    break
        
        # ---- BATCHED GENERATION LOOP ----
        decoder_start_token_id = getattr(config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(tokenizer, "bos_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(tokenizer, "cls_token_id", None)
        if decoder_start_token_id is None:
            raise ValueError(
                "Could not determine decoder_start_token_id. "
                "Please ensure your model config or tokenizer provides 'decoder_start_token_id', 'bos_token_id', or 'cls_token_id'."
            )
        decoder_input_ids = np.full((batch_size, 1), int(decoder_start_token_id), dtype=np.int64)
        
        eos_id = getattr(config, 'eos_token_id', None) or getattr(tokenizer, 'eos_token_id', None) or getattr(tokenizer, 'pad_token_id', -1)
        if eos_id is None or eos_id == -1:
            logging.warning("Could not determine a valid EOS token ID. Generation may not stop correctly.")

        generated_tokens = [[] for _ in range(batch_size)]
        unfinished_sequences = np.ones(batch_size, dtype=bool)
        use_cache_branch = np.array([False], dtype=np.bool_)
        encoder_past_frozen = False
        logits_output_name = output_infos[0].name

        for _ in range(max_generation_length):
            if not np.any(unfinished_sequences): break

            merged_past_kv = {**encoder_past_saved, **decoder_past_saved}
            decoder_inputs = {'input_ids': decoder_input_ids, 'encoder_attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states, 'use_cache_branch': use_cache_branch, **merged_past_kv}
            final_decoder_inputs = {k: v for k, v in decoder_inputs.items() if k in input_names}
            
            decoder_outputs = decoder_sess.run(None, final_decoder_inputs)
            output_map = {name.name: arr for name, arr in zip(output_infos, decoder_outputs)}

            logits = output_map.get(logits_output_name)
            if logits is None: raise RuntimeError(f"Could not find logits output '{logits_output_name}' in decoder outputs.")
            
            next_token_ids = np.argmax(logits[:, -1, :], axis=-1)

            # Update generated tokens for unfinished sequences
            for i in range(batch_size):
                if unfinished_sequences[i]:
                    token_id = next_token_ids[i]
                    if token_id == eos_id:
                        unfinished_sequences[i] = False
                    else:
                        generated_tokens[i].append(token_id)
            
            # Update KV caches
            for name, value in output_map.items():
                if 'present' in name and (past_name := present_to_past.get(name)):
                    if '.encoder.' in past_name:
                        if not encoder_past_frozen: encoder_past_saved[past_name] = value
                    elif past_name in decoder_past_saved: decoder_past_saved[past_name] = value

            decoder_input_ids = next_token_ids.reshape((batch_size, 1)).astype(np.int64)
            use_cache_branch[0] = True
            if not encoder_past_frozen: encoder_past_frozen = True
        
        # ---- BATCH DECODE AND SCORE CALCULATION ----
        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for prediction, ground_truth in zip(predictions, ground_truths):
            if 'wer' in scores and wer: scores['wer'].append(wer(ground_truth, prediction))
            if 'cer' in scores and cer: scores['cer'].append(cer(ground_truth, prediction))
            if 'accuracy' in scores: scores['accuracy'].append(1.0 if prediction.strip() == ground_truth.strip() else 0.0)
            if 'bleu' in scores and 'bleu' in metric_calculators: scores['bleu'].append(metric_calculators['bleu'].compute(predictions=[prediction], references=[[ground_truth]])['bleu'])
            if 'rouge' in metric_calculators and any(r in scores for r in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']):
                rouge_results = metric_calculators['rouge'].compute(predictions=[prediction], references=[ground_truth])
                if 'rouge1' in scores: scores['rouge1'].append(rouge_results['rouge1'])
                if 'rouge2' in scores: scores['rouge2'].append(rouge_results['rouge2'])
                if 'rougeL' in scores: scores['rougeL'].append(rouge_results['rougeL'])
                if 'rougeLsum' in scores: scores['rougeLsum'].append(rouge_results['rougeLsum'])

    except Exception as e:
        logging.error(f"  Benchmark failed for the batch: {e}", exc_info=True)
        # Populate scores with the worst possible value if the whole batch fails
        num_failed = len(samples) - len(scores[expanded_metrics[0]])
        for _ in range(num_failed):
            for metric in scores.keys(): scores[metric].append(1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    avg_scores = {metric: float(np.mean(score_list)) if score_list else (1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0) for metric, score_list in scores.items()}
    return avg_scores, elapsed_time


def load_samples_from_jsonl(path: Path) -> List[Dict]:
    """Loads samples from a JSONL file."""
    if not path.exists(): raise FileNotFoundError(f"Samples file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    if not samples: raise ValueError(f"No valid samples found in {path}.")
    return samples


def get_filesize_mb(*paths: Path) -> float:
    """Calculates the total size of one or more files in megabytes."""
    total_bytes = sum(os.path.getsize(p) for p in paths if p and p.exists())
    return total_bytes / (1024 * 1024)


def find_optimal_exclusions(
    model_to_search_path: Path,
    nodes_by_type: Dict[str, List[str]],
    part_name: str,
    fixed_encoder_path: Path,
    fixed_decoder_path: Path,
    baseline_score: float,
    reference_score: float,
    run_benchmark_args: Dict[str, Any],
    cfg: Dict[str, Any],
    quant_type: QuantType,
    quant_extra_options: Dict[str, Any],
    minimize_metric: bool,
    primary_metric: str,
    quant_test_dir: Path,
) -> List[str]:
    """
    Runs the two-stage search process for a given model component (encoder or decoder).
    
    Returns:
        A list of node names identified for exclusion.
    """
    strategy_stage1 = cfg["strategy_stage1"]
    strategy_stage2 = cfg["strategy_stage2"]
    candidate_op_types = cfg["candidate_op_types"]
    max_nodes_to_exclude = cfg["max_nodes_to_exclude"]
    target = cfg["target"]

    logging.info(f"--- STAGE 1: Discovering Sensitive Operation Types for {part_name.upper()} (Strategy: {strategy_stage1}) ---")

    tipping_point_op_type = None
    tipping_point_score = baseline_score
    stage2_candidate_nodes = []

    if strategy_stage1 == 'first':
        cumulative_exclusion_nodes = []
        last_score = baseline_score
        ops = list(candidate_op_types)
        pbar_s1 = tqdm(ops, desc=f"Stage 1/{part_name}", unit="op", total=len(ops))
        for op_type in pbar_s1:
            nodes_to_add = nodes_by_type.get(op_type, [])
            if not nodes_to_add: continue

            pbar_s1.set_description(f"Stage 1/{part_name} (Testing {op_type}, {len(nodes_to_add)} nodes)")
            current_test_exclusion_list = cumulative_exclusion_nodes + nodes_to_add

            fd, temp_model_path_str = tempfile.mkstemp(suffix=".onnx", dir=quant_test_dir)
            os.close(fd)
            temp_model_path = Path(temp_model_path_str)
            try:
                quantize_dynamic(model_to_search_path, temp_model_path, weight_type=quant_type, nodes_to_exclude=current_test_exclusion_list, extra_options=quant_extra_options)
                if part_name == "decoder":
                    current_metrics, _ = run_benchmark(fixed_encoder_path, temp_model_path, **run_benchmark_args)
                else: # encoder
                    current_metrics, _ = run_benchmark(temp_model_path, fixed_decoder_path, **run_benchmark_args)
            finally:
                if temp_model_path.exists():
                    temp_model_path.unlink()
            
            current_score = current_metrics[primary_metric]
            pbar_s1.set_postfix({primary_metric: f"{current_score:.4f}"})

            cumulative_exclusion_nodes.extend(nodes_to_add)

            # Early stop
            fp32_level_met = (current_score <= reference_score) if minimize_metric else (current_score >= reference_score)
            if fp32_level_met:
                logging.info(f"  ✅ Early stop! FP32 performance level reached ({primary_metric.upper()}: {current_score:.4f} vs FP32 Ref: {reference_score:.4f}).")
                tipping_point_op_type, tipping_point_score = op_type, current_score
                stage2_candidate_nodes = list(cumulative_exclusion_nodes)
                break

            is_improvement = (current_score < last_score) if minimize_metric else (current_score > last_score)
            if is_improvement:
                logging.info(f"  ✅ Tipping point! '{op_type}' improved {primary_metric.upper()} from {last_score:.4f} to {current_score:.4f}.")
                tipping_point_op_type, tipping_point_score = op_type, current_score
                stage2_candidate_nodes = list(cumulative_exclusion_nodes)
                break
            last_score = current_score
        pbar_s1.close()

    elif strategy_stage1 == 'best':
        best_overall_score = baseline_score
        best_op_type, best_cumulative_nodes = None, []
        cumulative_exclusion_nodes = []

        ops = list(candidate_op_types)
        pbar_s1 = tqdm(ops, desc=f"Stage 1/{part_name}", unit="op", total=len(ops))
        for op_type in pbar_s1:
            nodes_to_add = nodes_by_type.get(op_type, [])
            if not nodes_to_add: continue

            pbar_s1.set_description(f"Stage 1/{part_name} (Testing {op_type}, {len(nodes_to_add)} nodes)")
            current_test_exclusion_list = cumulative_exclusion_nodes + nodes_to_add

            fd, temp_model_path_str = tempfile.mkstemp(suffix=".onnx", dir=quant_test_dir)
            os.close(fd)
            temp_model_path = Path(temp_model_path_str)
            try:
                quantize_dynamic(model_to_search_path, temp_model_path, weight_type=quant_type, nodes_to_exclude=current_test_exclusion_list, extra_options=quant_extra_options)

                if part_name == "decoder":
                    current_metrics, _ = run_benchmark(fixed_encoder_path, temp_model_path, **run_benchmark_args)
                else: # encoder
                    current_metrics, _ = run_benchmark(temp_model_path, fixed_decoder_path, **run_benchmark_args)
            
            finally:
                if temp_model_path.exists():
                    temp_model_path.unlink()
            
            current_score = current_metrics[primary_metric]
            pbar_s1.set_postfix({primary_metric: f"{current_score:.4f}"})

            cumulative_exclusion_nodes.extend(nodes_to_add)

            # Early stop
            fp32_level_met = (current_score <= reference_score) if minimize_metric else (current_score >= reference_score)
            if fp32_level_met:
                logging.info(f"  ✅ Early stop! FP32 performance level reached ({primary_metric.upper()}: {current_score:.4f} vs FP32 Ref: {reference_score:.4f}).")
                tipping_point_op_type, tipping_point_score = op_type, current_score
                stage2_candidate_nodes = list(cumulative_exclusion_nodes)
                break

            is_best_so_far = (current_score < best_overall_score) if minimize_metric else (current_score > best_overall_score)
            if is_best_so_far:
                logging.info(f"  Found new best score with '{op_type}'. {primary_metric.upper()} improved from {best_overall_score:.4f} to {current_score:.4f}.")
                best_overall_score, best_op_type = current_score, op_type
                best_cumulative_nodes = list(cumulative_exclusion_nodes)
        pbar_s1.close()

        if not tipping_point_op_type and best_op_type: # Only if early stop didn't trigger
            logging.info(f"  ✅ Best tipping point found with op type '{best_op_type}'.")
            tipping_point_op_type, tipping_point_score = best_op_type, best_overall_score
            stage2_candidate_nodes = best_cumulative_nodes
    
    elif strategy_stage1 == 'percent':
        performance_gap = abs(reference_score - baseline_score)
        improvement_needed = performance_gap * target
        target_score = (baseline_score - improvement_needed) if minimize_metric else (baseline_score + improvement_needed)
        logging.info(f"Target score to reach: {target_score:.4f} ({target:.0%} of the gap between baseline and reference)")

        cumulative_exclusion_nodes = []
        ops = list(candidate_op_types)
        pbar_s1 = tqdm(ops, desc=f"Stage 1/{part_name}", unit="op", total=len(ops))
        for op_type in pbar_s1:
            nodes_to_add = nodes_by_type.get(op_type, [])
            if not nodes_to_add: continue

            pbar_s1.set_description(f"Stage 1/{part_name} (Testing {op_type}, {len(nodes_to_add)} nodes)")
            current_test_exclusion_list = cumulative_exclusion_nodes + nodes_to_add
            
            fd, temp_model_path_str = tempfile.mkstemp(suffix=".onnx", dir=quant_test_dir)
            os.close(fd)
            temp_model_path = Path(temp_model_path_str)
            try:
                quantize_dynamic(model_to_search_path, temp_model_path, weight_type=quant_type, nodes_to_exclude=current_test_exclusion_list, extra_options=quant_extra_options)
                
                if part_name == "decoder":
                    current_metrics, _ = run_benchmark(fixed_encoder_path, temp_model_path, **run_benchmark_args)
                else: # encoder
                    current_metrics, _ = run_benchmark(temp_model_path, fixed_decoder_path, **run_benchmark_args)

            finally:
                if temp_model_path.exists():
                    temp_model_path.unlink()
            
            current_score = current_metrics[primary_metric]
            pbar_s1.set_postfix({primary_metric: f"{current_score:.4f}"})
            
            cumulative_exclusion_nodes.extend(nodes_to_add)

            # Early stop
            fp32_level_met = (current_score <= reference_score) if minimize_metric else (current_score >= reference_score)
            if fp32_level_met:
                logging.info(f"  ✅ Early stop! FP32 performance level reached ({primary_metric.upper()}: {current_score:.4f} vs FP32 Ref: {reference_score:.4f}).")
                tipping_point_op_type, tipping_point_score = op_type, current_score
                stage2_candidate_nodes = list(cumulative_exclusion_nodes)
                break

            target_met = (current_score <= target_score) if minimize_metric else (current_score >= target_score)
            if target_met:
                logging.info(f"  ✅ Target met! '{op_type}' reached the target score of {target_score:.4f}.")
                tipping_point_op_type, tipping_point_score = op_type, current_score
                stage2_candidate_nodes = list(cumulative_exclusion_nodes)
                break
        pbar_s1.close()

    if not tipping_point_op_type:
        logging.error(f"Stage 1 for {part_name.upper()} failed: No cumulative exclusion improved the baseline {primary_metric.upper()} using strategy '{strategy_stage1}'.")
        return []

    logging.info(f"\n✅ Stage 1 for {part_name.upper()} Complete. Tipping point: '{tipping_point_op_type}'. Pool for Stage 2 has {len(stage2_candidate_nodes)} nodes.\n")

    logging.info(f"--- STAGE 2: Pruning nodes from the {len(stage2_candidate_nodes)} candidates for {part_name.upper()} (Strategy: {strategy_stage2}) ---")
    if max_nodes_to_exclude is not None:
        logging.info(f"Will stop pruning if the exclusion list has {max_nodes_to_exclude} or fewer nodes.")
    
    best_exclusion_list = list(stage2_candidate_nodes)
    current_best_score = tipping_point_score
    logging.info(f"Initial {primary_metric.upper()} with all nodes: {current_best_score:.4f}")

    changed = True
    pass_num = 0
    num_initial_candidates = len(stage2_candidate_nodes)
    while changed and (max_nodes_to_exclude is None or len(best_exclusion_list) > max_nodes_to_exclude):
        pass_num += 1
        changed = False
        nodes_to_test = list(best_exclusion_list)
        if len(nodes_to_test) == 0:
            break

        logging.info(f"--- Pruning Pass #{pass_num} ({len(nodes_to_test)} nodes left) ---")

        pbar_s2 = tqdm(nodes_to_test, desc=f"Stage 2/{part_name}", unit="node", total=num_initial_candidates, initial=num_initial_candidates - len(nodes_to_test), leave=False)
        for node in pbar_s2:
            short_node_name = node.split('/')[-1]
            pbar_s2.set_description(f"Stage 2/{part_name} (Test remove: {short_node_name})")
            best_exclusion_list.remove(node)
            
            fd, temp_model_path_str = tempfile.mkstemp(suffix=".onnx", dir=quant_test_dir)
            os.close(fd)
            temp_model_path = Path(temp_model_path_str)
            try:
                # Use the modified list for quantization
                quantize_dynamic(model_to_search_path, temp_model_path, weight_type=quant_type, nodes_to_exclude=best_exclusion_list, extra_options=quant_extra_options)
                
                if part_name == "decoder":
                    metrics, _ = run_benchmark(fixed_encoder_path, temp_model_path, **run_benchmark_args)
                else: # encoder
                    metrics, _ = run_benchmark(temp_model_path, fixed_decoder_path, **run_benchmark_args)
            finally:
                # Ensure the temporary file is deleted
                if temp_model_path.exists():
                    temp_model_path.unlink()

            score = metrics[primary_metric]
            logging.info(f"  Test removing node '{node}' from {part_name.upper()} -> {primary_metric.upper()}: {score:.4f}")
            
            can_be_removed = False
            if strategy_stage2 == 'relaxed':
                can_be_removed = (score <= tipping_point_score + 1e-6) if minimize_metric else (score >= tipping_point_score - 1e-6)
            elif strategy_stage2 == 'strict':
                can_be_removed = (score <= current_best_score + 1e-6) if minimize_metric else (score >= current_best_score - 1e-6)

            if can_be_removed:
                logging.info(f"    ✅ Node '{node}' removed. New {primary_metric.upper()}: {score:.4f}")
                current_best_score = score
                changed = True
                pbar_s2.close()
                break
            else:
                # Add the node back if it cannot be removed
                best_exclusion_list.append(node)
        if not changed:
            pbar_s2.close()

    if max_nodes_to_exclude is not None and len(best_exclusion_list) > max_nodes_to_exclude:
         logging.warning(f"Pruning finished, but the {part_name.upper()} exclusion list ({len(best_exclusion_list)} nodes) is larger than the specified max_nodes_to_exclude ({max_nodes_to_exclude}).")

    logging.info(f"\n--- {part_name.upper()} Pruning Complete ---")
    if not best_exclusion_list:
        logging.warning(f"Could not find any nodes to exclude from the {part_name.upper()} that improved accuracy.")
    else:
        logging.info(f"Found minimal exclusion list for {part_name.upper()} with {len(best_exclusion_list)} nodes, achieving a {primary_metric.upper()} of {current_best_score:.4f}")

    return best_exclusion_list


if __name__ == "__main__":
    script_start_time = time.perf_counter()
    # Get the script's directory to resolve default paths
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Discover nodes to exclude from quantization for encoder-decoder ONNX models.")
    parser.add_argument("--config", "-c", type=str, default=str(script_dir / "config_quant.json"), help="Path to JSON config file (overrides defaults).")
    parser.add_argument("--results", type=str, default=str(script_dir / "results.jsonl"), help="Path to export the final exclusion lists as a JSONL file.")
    parser.add_argument("--export_final", type=str, help="Path to a directory to export the final quantized ONNX models with the discovered exclusions.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

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
    MODEL_DIR, ONNX_DIR = Path(cfg["model_dir"]), Path(cfg["onnx_dir"])
    QUANT_TEST_DIR = Path(cfg["quant_test_dir"])
    samples_jsonl = Path(cfg["samples_jsonl"])
    if not samples_jsonl.is_absolute():
        samples_jsonl = script_dir / samples_jsonl
    PRIMARY_METRIC, METRICS = cfg["primary_metric"], cfg["metrics"]
    EXECUTION_PROVIDER, MAX_GENERATION_LENGTH = cfg["execution_provider"], cfg["max_generation_length"]
    QUANT_TYPE_STR, SEARCH_TARGET = cfg["quant_type"], cfg["search_target"]
    TARGET = cfg["target"]
    DEVICE = "cpu" if EXECUTION_PROVIDER == "CPUExecutionProvider" else "cuda"

    logging.info(f"Search Target: '{SEARCH_TARGET}'")
    logging.info(f"Using Stage 1 strategy: '{cfg['strategy_stage1']}'")
    if cfg['strategy_stage1'] == "percent":
        if TARGET is None or not (0.0 <= TARGET <= 1.0):
            raise ValueError("For 'percent' strategy, 'target' must be set in the config file as a float between 0.0 and 1.0.")

    if QUANT_TYPE_STR == "QUInt8":
        quant_type, quant_suffix = QuantType.QUInt8, "quint8"
    elif QUANT_TYPE_STR == "QInt8":
        quant_type, quant_suffix = QuantType.QInt8, "qint8"
    else:
        raise ValueError(f"Unsupported 'quant_type' in config: '{QUANT_TYPE_STR}'. Must be 'QUInt8' or 'QInt8'.")

    expanded_primary_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'] if 'rouge' in METRICS else []
    expanded_primary_metrics.extend([m for m in METRICS if m != 'rouge'])
    if PRIMARY_METRIC not in expanded_primary_metrics:
        raise ValueError(f"Primary metric '{PRIMARY_METRIC}' must be one of the tested metrics: {expanded_primary_metrics}")

    minimize_metric = PRIMARY_METRIC not in HIGHER_IS_BETTER_METRICS

    fp32_encoder_path, fp32_decoder_path = ONNX_DIR / cfg["fp32_encoder"], ONNX_DIR / cfg["fp32_decoder"]
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    QUANT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    if cfg["model_reference"] == "safetensors":
        if not fp32_encoder_path.exists() or not fp32_decoder_path.exists():
            logging.info(f"Exporting safetensors model from '{MODEL_DIR}' to ONNX in '{ONNX_DIR}'...")
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
    
    if not fp32_encoder_path.exists() or not fp32_decoder_path.exists():
        raise FileNotFoundError(f"Missing FP32 ONNX model. Check paths in config: {fp32_encoder_path}, {fp32_decoder_path} or set 'model_reference' to 'safetensors'.")

    tokenizer, config = AutoTokenizer.from_pretrained(str(MODEL_DIR)), AutoConfig.from_pretrained(str(MODEL_DIR))
    samples = load_samples_from_jsonl(samples_jsonl)
    logging.info(f"Loaded {len(samples)} sample(s) from {samples_jsonl} for benchmarking.\n")
    
    quant_extra_options = {"EnableSubgraph": cfg["enable_subgraph"]}
    
    run_benchmark_args = {
        "tokenizer": tokenizer, "config": config, "samples": samples,
        "metrics": METRICS, "execution_provider": EXECUTION_PROVIDER,
        "max_generation_length": MAX_GENERATION_LENGTH
    }

    logging.info("🔬 Benchmarking FP32 ONNX Model (Reference)...")
    reference_metrics, reference_time = run_benchmark(fp32_encoder_path, fp32_decoder_path, **run_benchmark_args)
    REFERENCE_SCORE = reference_metrics[PRIMARY_METRIC]
    fp32_filesize = get_filesize_mb(fp32_encoder_path, fp32_decoder_path)
    logging.info(f"   ✅ Reference (FP32) Metrics: {json.dumps({k: round(v, 4) for k, v in reference_metrics.items()})}")
    logging.info(f"   ✅ Time: {reference_time:.4f}s, Size: {fp32_filesize:.2f}MB")


    logging.info("🔬 Benchmarking Fully Quantized Model (Baseline)...")
    baseline_quant_encoder_path = QUANT_TEST_DIR / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic.onnx")
    baseline_quant_decoder_path = QUANT_TEST_DIR / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic.onnx")
    quantize_dynamic(fp32_encoder_path, baseline_quant_encoder_path, weight_type=quant_type)
    quantize_dynamic(fp32_decoder_path, baseline_quant_decoder_path, weight_type=quant_type, extra_options=quant_extra_options)
    baseline_metrics, baseline_time = run_benchmark(baseline_quant_encoder_path, baseline_quant_decoder_path, **run_benchmark_args)
    BASELINE_SCORE = baseline_metrics[PRIMARY_METRIC]
    baseline_quant_filesize = get_filesize_mb(baseline_quant_encoder_path, baseline_quant_decoder_path)
    logging.info(f"   ✅ Baseline (Quantized) Metrics: {json.dumps({k: round(v, 4) for k, v in baseline_metrics.items()})}")
    logging.info(f"   ✅ Time: {baseline_time:.4f}s, Size: {baseline_quant_filesize:.2f}MB\n")


    decoder_exclusion_list, encoder_exclusion_list = [], []
    
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
        logging.info(f"   ✅ Encoder Search Baseline {PRIMARY_METRIC.upper()}: {encoder_search_baseline_score:.4f}")

        # Check if the baseline already meets the reference performance
        fp32_level_met = (encoder_search_baseline_score <= REFERENCE_SCORE) if minimize_metric else (encoder_search_baseline_score >= REFERENCE_SCORE)

        if fp32_level_met:
            logging.info(f"✅ Encoder search baseline ({encoder_search_baseline_score:.4f}) already meets/exceeds the FP32 reference ({REFERENCE_SCORE:.4f}).")
            logging.info("Skipping node exclusion search for the encoder as it's not needed.")
            encoder_exclusion_list = []
        else:
            logging.info("Loading encoder model once to extract node names...")
            encoder_model_onnx = onnx.load(str(fp32_encoder_path))
            encoder_nodes_by_type = {op: [n.name for n in find_nodes_recursively(encoder_model_onnx.graph, [op]) if n.name] for op in cfg["candidate_op_types"]}

            encoder_exclusion_list = find_optimal_exclusions(
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
        logging.info(f"   ✅ Decoder Search Baseline {PRIMARY_METRIC.upper()}: {decoder_search_baseline_score:.4f}")

        # Check if the baseline already meets the reference performance
        fp32_level_met = (decoder_search_baseline_score <= REFERENCE_SCORE) if minimize_metric else (decoder_search_baseline_score >= REFERENCE_SCORE)

        if fp32_level_met:
            logging.info(f"✅ Decoder search baseline ({decoder_search_baseline_score:.4f}) already meets/exceeds the FP32 reference ({REFERENCE_SCORE:.4f}).")
            logging.info("Skipping node exclusion search for the decoder as it's not needed.")
            decoder_exclusion_list = []
        else:
            logging.info("Loading decoder model once to extract node names...")
            decoder_model_onnx = onnx.load(str(fp32_decoder_path))
            decoder_nodes_by_type = {op: [n.name for n in find_nodes_recursively(decoder_model_onnx.graph, [op]) if n.name] for op in cfg["candidate_op_types"]}

            decoder_exclusion_list = find_optimal_exclusions(
                model_to_search_path=fp32_decoder_path,
                nodes_by_type=decoder_nodes_by_type,
                part_name="decoder",
                fixed_encoder_path=fp32_encoder_path,  # Use FP32 encoder to isolate decoder issues
                fixed_decoder_path=None,  # Not used when searching decoder
                baseline_score=decoder_search_baseline_score,
                **search_args
            )

    # --- Run a final benchmark with the optimal exclusion lists ---
    logging.info("\n🔬 Benchmarking final configuration with optimal exclusions...")
    final_quant_encoder_path = QUANT_TEST_DIR / (fp32_encoder_path.stem + f".{quant_suffix}_dynamic_final_benchmark.onnx")
    final_quant_decoder_path = QUANT_TEST_DIR / (fp32_decoder_path.stem + f".{quant_suffix}_dynamic_final_benchmark.onnx")
    final_metrics, final_time, final_quant_filesize = {}, 0.0, 0.0
    try:
        # Quantize encoder with its final exclusion list
        quantize_dynamic(
            fp32_encoder_path,
            final_quant_encoder_path,
            weight_type=quant_type,
            nodes_to_exclude=encoder_exclusion_list
        )
        # Quantize decoder with its final exclusion list
        quantize_dynamic(
            fp32_decoder_path,
            final_quant_decoder_path,
            weight_type=quant_type,
            nodes_to_exclude=decoder_exclusion_list,
            extra_options=quant_extra_options
        )
        # Run the benchmark on the final configuration
        final_metrics, final_time = run_benchmark(final_quant_encoder_path, final_quant_decoder_path, **run_benchmark_args)
        final_quant_filesize = get_filesize_mb(final_quant_encoder_path, final_quant_decoder_path)
        delta_baseline = final_metrics[PRIMARY_METRIC] - REFERENCE_SCORE
        logging.info(f"   ✅ Final Optimized Metrics: {json.dumps({k: round(v, 4) for k, v in final_metrics.items()})}")
        logging.info(f"   ✅ Time: {final_time:.4f}s, Size: {final_quant_filesize:.2f}MB")
        logging.info(f"   ✅ Delta from FP32 Baseline ({PRIMARY_METRIC.upper()}): {delta_baseline:+.4f}")

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
            quantize_dynamic(fp32_encoder_path, final_quant_encoder_path, weight_type=quant_type, nodes_to_exclude=encoder_exclusion_list)
            logging.info(f"  -> Saved to {final_quant_encoder_path}")

            logging.info(f"Quantizing decoder with {len(decoder_exclusion_list)} excluded nodes...")
            quantize_dynamic(fp32_decoder_path, final_quant_decoder_path, weight_type=quant_type, nodes_to_exclude=decoder_exclusion_list, extra_options=quant_extra_options)
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