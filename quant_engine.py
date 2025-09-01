import logging
import tempfile
import os
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Local imports
from quant_utils import _extract_sample_fields, _load_state, _save_state_atomic, _remove_state, SuppressQuantFilter

# Attach filter to ONNX Runtime and root logger to suppress those noisy warnings
logging.getLogger('onnxruntime').addFilter(SuppressQuantFilter())
logging.getLogger().addFilter(SuppressQuantFilter())

# ==== CONSTANTS ====
HIGHER_IS_BETTER_METRICS = {"accuracy", "bleu", "rouge", "rouge1", "rouge2", "rougeL", "rougeLsum"}

def find_logits_output(output_infos: List[Any], vocab_size: Optional[int] = None) -> str:
    """
    Heuristic to locate logits output name from decoder session outputs.
    Preferred strategies (in order):
     - name contains 'logit' / 'lm_' / 'scores'
     - last dim equals vocab_size (if provided and numeric)
     - fallback to first output with warning
    """
    # First pass: name heuristics
    for out in output_infos:
        lname = out.name.lower()
        if "logit" in lname or "lm_" in lname or "scores" in lname:
            logging.debug(f"Selected logits output by name heuristic: {out.name}")
            return out.name

    # Second pass: shape heuristics
    for out in output_infos:
        shape = getattr(out, "shape", None)
        if not shape:
            # Some ORT versions expose .shape as list or tuple; if not, skip
            continue
        try:
            last_dim = shape[-1]
            if isinstance(last_dim, int) and vocab_size and last_dim == vocab_size:
                logging.debug(f"Selected logits output by shape match: {out.name}")
                return out.name
        except Exception:
            continue

    # Last resort: first output
    logging.warning("Could not identify logits output by name/shape heuristics; falling back to the first output.")
    return output_infos[0].name


def map_present_to_past(decoder_inputs: List[Any], decoder_outputs: List[Any]) -> Dict[str, str]:
    """
    Build a mapping from 'present' output names to corresponding 'past' input names.
    Uses heuristics: name replacements, substring matches, and existence in input names.
    """
    input_names = {inp.name for inp in decoder_inputs}
    mapping = {}
    for out in decoder_outputs:
        oname = out.name
        lname = oname.lower()
        if "present" not in lname and ("key" not in lname and "value" not in lname):
            continue
        # Try several replacement patterns
        candidates = []
        # direct replacements
        candidates.append(oname.replace("present", "past"))
        candidates.append(oname.replace("present", "past_key_values"))
        candidates.append(oname.replace("present", "past_key_value"))
        candidates.append(oname + "_past")
        # remove dots/underscores permutations
        candidates.append(oname.replace(".", "_") + "_past")
        candidates.append(oname.replace("present", "past_key_values").replace(".", "_"))
        # try simpler containing tokens
        base = oname
        candidates.extend([c for c in [base, base + "_past", "past_" + base] if c])
        # find first candidate that exists
        for c in candidates:
            if c in input_names:
                mapping[oname] = c
                break
        # fallback: try matching by token substrings (e.g., both contain 'encoder' or 'decoder')
        if oname not in mapping:
            for inp_name in input_names:
                if oname.endswith(inp_name) or inp_name.endswith(oname):
                    mapping[oname] = inp_name
                    break
    logging.debug(f"Present->Past mapping discovered: {mapping}")
    return mapping


def build_initial_past_kv(decoder_inputs: List[Any], batch_size: int, encoder_seq_len: int, past_decoder_seq_len: int = 1) -> Dict[str, np.ndarray]:
    """
    Construct zero-filled arrays for 'past' inputs. Heuristics for symbolic shapes:
     - first dim -> batch_size
     - dims containing 'encoder' -> encoder_seq_len
     - dims containing 'decoder' -> past_decoder_seq_len
     - unknown dims -> 1
    """
    past_kv = {}
    for inp in decoder_inputs:
        name = inp.name
        lname = name.lower()
        if "past" not in lname and "key" not in lname and "value" not in lname:
            continue
        raw_shape = getattr(inp, "shape", None)
        if not raw_shape:
            # If no shape metadata, attempt a reasonable default: (batch, 1, hidden) -> unknown hidden dim -> use 1
            past_kv[name] = np.zeros((batch_size, past_decoder_seq_len, 1), dtype=np.float32)
            continue
        shape = []
        for idx, dim in enumerate(raw_shape):
            try:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    # symbolic (str) or None
                    if idx == 0:
                        shape.append(batch_size)
                    elif 'encoder' in lname:
                        shape.append(encoder_seq_len)
                    elif 'decoder' in lname:
                        shape.append(past_decoder_seq_len)
                    else:
                        # default small size to avoid large allocations
                        shape.append(1)
            except Exception:
                shape.append(1)
        try:
            past_kv[name] = np.zeros(tuple(shape), dtype=np.float32)
        except Exception:
            # final fallback: small shape
            past_kv[name] = np.zeros((batch_size, past_decoder_seq_len, 1), dtype=np.float32)
    return past_kv


def run_benchmark(
    encoder_path: Path,
    decoder_path: Path,
    tokenizer: AutoTokenizer,
    config: AutoConfig,
    samples: List[Dict],
    metrics: List[str],
    execution_provider: str,
    max_generation_length: int,
    quant_test_dir: Path
) -> Tuple[Dict[str, float], float]:
    """
    Runs inference on an encoder-decoder ONNX model pair and returns a dictionary of averaged metrics and the inference time.
    Inference is performed in a single batch for all samples to improve performance.

    NOTE: This function expects encoder_path and decoder_path to point to ONNX files.
    """
    # Conditionally import evaluation libraries based on requested metrics
    load_metric = None
    if any(m in metrics for m in ["bleu", "rouge"]):
        from evaluate import load as load_metric

    wer, cer = None, None
    if any(m in metrics for m in ["wer", "cer"]):
        from jiwer import wer, cer

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

    # Initialize elapsed_time for inference
    inference_time = 0.0
    try:
        # ==== START TIMER FOR MODEL LOADING ====
        load_start_time = time.perf_counter()
        
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 3
        encoder_sess = ort.InferenceSession(str(encoder_path), providers=[execution_provider], sess_options=sess_opt)
        decoder_sess = ort.InferenceSession(str(decoder_path), providers=[execution_provider], sess_options=sess_opt)
        
        # ==== STOP TIMER FOR MODEL LOADING AND PRINT ====
        load_end_time = time.perf_counter()
        loading_time = load_end_time - load_start_time
        logging.info(f"🕑 Model loading time: {loading_time:.4f} seconds")

    except Exception as e:
        logging.error(f"Failed to create ONNX Runtime sessions: {e}")
        return {metric: 1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0 for metric in expanded_metrics}, 0.0

    try:
        # ---- BATCH PREPARATION ----
        input_prompts, ground_truths = zip(*[_extract_sample_fields(s) for s in samples])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(list(input_prompts), return_tensors="np", padding=True, truncation=True, max_length=getattr(config, "max_length", 512))
        input_ids = inputs['input_ids'].astype(np.int64)
        attention_mask = inputs['attention_mask'].astype(np.int64)
        batch_size = input_ids.shape[0]

        # ==== START TIMER FOR INFERENCE ====
        inference_start_time = time.perf_counter()

        # Encoder forward
        encoder_hidden_states = encoder_sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        
        # ---- DECODER SETUP ----
        decoder_input_details = decoder_sess.get_inputs()
        encoder_seq_len = encoder_hidden_states.shape[1]
        past_decoder_seq_len = 1

        # Build initial past_kv (only for inputs that look like past/key/value)
        past_kv = build_initial_past_kv(decoder_input_details, batch_size, encoder_seq_len, past_decoder_seq_len)
        encoder_past_saved = {k: v for k, v in past_kv.items() if '.encoder.' in k or 'encoder' in k}
        decoder_past_saved = {k: v for k, v in past_kv.items() if '.decoder.' in k or 'decoder' in k}

        # Map present outputs to past inputs more robustly
        output_infos = decoder_sess.get_outputs()
        present_to_past = map_present_to_past(decoder_input_details, output_infos)

        # Detect logits output name using heuristics and vocab size if available
        vocab_size = getattr(config, "vocab_size", None)
        logits_output_name = find_logits_output(output_infos, vocab_size=vocab_size)

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
        use_cache_branch = np.array([False], dtype=bool)
        encoder_past_frozen = False

        # We'll convert outputs to a map by name after running session
        for _ in range(max_generation_length):
            if not np.any(unfinished_sequences): break

            merged_past_kv = {**encoder_past_saved, **decoder_past_saved}
            decoder_inputs = {'input_ids': decoder_input_ids, 'encoder_attention_mask': attention_mask, 'encoder_hidden_states': encoder_hidden_states, 'use_cache_branch': use_cache_branch, **merged_past_kv}
            # Finalize inputs to those actually present in the session
            final_decoder_inputs = {k: v for k, v in decoder_inputs.items() if k in {inp.name for inp in decoder_input_details}}
            
            decoder_outputs = decoder_sess.run(None, final_decoder_inputs)
            output_map = {name.name: arr for name, arr in zip(output_infos, decoder_outputs)}

            logits = output_map.get(logits_output_name)
            if logits is None:
                # Try to find any output that looks like logits by shape
                found_logits = None
                for k, arr in output_map.items():
                    if arr.ndim >= 2:
                        # logits shape is usually (batch, seq_len, vocab)
                        if arr.shape[-1] == vocab_size if vocab_size else True:
                            found_logits = arr
                            logits_output_name = k
                            break
                if found_logits is None:
                    raise RuntimeError(f"Could not find logits output '{logits_output_name}' in decoder outputs. Available outputs: {list(output_map.keys())}")
                logits = found_logits

            next_token_ids = np.argmax(logits[:, -1, :], axis=-1)

            # Update generated tokens for unfinished sequences
            for i in range(batch_size):
                if unfinished_sequences[i]:
                    token_id = int(next_token_ids[i])
                    if token_id == eos_id:
                        unfinished_sequences[i] = False
                    else:
                        generated_tokens[i].append(token_id)
            
            # Update KV caches using the present_to_past mapping
            for name, value in output_map.items():
                if name in present_to_past:
                    past_name = present_to_past[name]
                    if 'encoder' in past_name and not encoder_past_frozen:
                        encoder_past_saved[past_name] = value
                    elif 'decoder' in past_name or past_name in decoder_past_saved:
                        decoder_past_saved[past_name] = value

            decoder_input_ids = next_token_ids.reshape((batch_size, 1)).astype(np.int64)
            use_cache_branch[0] = True
            if not encoder_past_frozen: encoder_past_frozen = True
        
        # ==== STOP TIMER FOR INFERENCE ====
        inference_end_time = time.perf_counter()
        inference_time = inference_end_time - inference_start_time

        # ---- BATCH DECODE AND SCORE CALCULATION ----
        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # --- Print predictions and ground truths for inspection ---
        logging.info("Sample predictions vs. ground truths:")
        for i, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
            logging.info(f"Sample {i+1}:")
            logging.info(f"Prediction: {prediction}")
            logging.info(f"Ground truth: {ground_truth}")

        for prediction, ground_truth in zip(predictions, ground_truths):
            if 'wer' in scores and wer: scores['wer'].append(wer(ground_truth, prediction))
            if 'cer' in scores and cer: scores['cer'].append(cer(ground_truth, prediction))
            if 'accuracy' in scores: scores['accuracy'].append(1.0 if prediction.strip() == ground_truth.strip() else 0.0)
            if 'bleu' in scores and 'bleu' in metric_calculators:
                scores['bleu'].append(metric_calculators['bleu'].compute(predictions=[prediction], references=[[ground_truth]])['bleu'])
            # rouge handling
            if 'rouge' in metric_calculators and any(r in scores for r in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']):
                rouge_results = metric_calculators['rouge'].compute(predictions=[prediction], references=[ground_truth])
                if 'rouge1' in scores: scores['rouge1'].append(rouge_results.get('rouge1', 0.0))
                if 'rouge2' in scores: scores['rouge2'].append(rouge_results.get('rouge2', 0.0))
                if 'rougeL' in scores: scores['rougeL'].append(rouge_results.get('rougeL', 0.0))
                if 'rougeLsum' in scores: scores['rougeLsum'].append(rouge_results.get('rougeLsum', 0.0))

    except Exception as e:
        logging.error(f"  Benchmark failed for the batch: {e}", exc_info=True)
        # Populate scores with the worst possible value if the whole batch fails
        num_failed = len(samples) - len(scores[expanded_metrics[0]])
        for _ in range(num_failed):
            for metric in scores.keys(): scores[metric].append(1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0)
    
    avg_scores = {metric: float(np.mean(score_list)) if score_list else (1.0 if metric not in HIGHER_IS_BETTER_METRICS else 0.0) for metric, score_list in scores.items()}
    return avg_scores, inference_time

def dynamic_quantization(
    model_to_search_path: Path,
    out_path: Path,
    quant_type: QuantType,
    nodes_to_exclude: List[str],
    extra_options: Dict[str, Any]
) -> None:
    q_args = {
        "weight_type": quant_type,
        "nodes_to_exclude": nodes_to_exclude
    }
    if extra_options:
        q_args["extra_options"] = extra_options
    try:
        quantize_dynamic(str(model_to_search_path), str(out_path), **q_args)
    except Exception as e:
        logging.error(f"Quantization failed for {model_to_search_path} -> {out_path}: {e}")
        raise
        
def _initialize_search_state(
    state_path: Path,
    part_name: str,
    candidate_op_types: List[str],
    baseline_score: float,
    resume_enabled: bool
) -> Dict[str, Any]:
    """Loads a previously saved state or initializes a new one."""
    state = _load_state(state_path) if resume_enabled else None
    
    if state and state.get("part") == part_name:
        saved_stage = int(state.get("stage", 1))
        logging.info(f"Resuming previous search for {part_name.upper()} from state file: {state_path} (stage {saved_stage})")
        
        initial_state = {
            "ops_remaining": state.get("ops_remaining", list(candidate_op_types)),
            "cumulative_exclusion_nodes": state.get("cumulative_exclusion_nodes", []),
            "per_op_results": state.get("per_op_results", []),
            "tipping_point_op_type": state.get("tipping_point_op_type"),
            "tipping_point_score": state.get("tipping_point_score", baseline_score),
            "stage2_candidate_nodes": state.get("stage2_candidate_nodes", []),
            "best_exclusion_list": state.get("best_exclusion_list", []),
            "current_best_score": state.get("current_best_score", baseline_score),
            "stage2_nodes_left": state.get("stage2_nodes_left", []),
            "stage1_skipped_due_to_resume": saved_stage == 2
        }
    else:
        initial_state = {
            "ops_remaining": list(candidate_op_types),
            "cumulative_exclusion_nodes": [],
            "per_op_results": [],
            "tipping_point_op_type": None,
            "tipping_point_score": baseline_score,
            "stage2_candidate_nodes": [],
            "best_exclusion_list": [],
            "current_best_score": baseline_score,
            "stage2_nodes_left": [],
            "stage1_skipped_due_to_resume": False
        }
        
    return initial_state

def _run_stage1_search(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    part_name: str,
    nodes_by_type: Dict[str, List[str]],
    baseline_score: float,
    reference_score: float,
    minimize_metric: bool,
    primary_metric: str,
    quant_and_benchmark: callable,
    persist_state: callable
) -> Dict[str, Any]:
    """Executes the Stage 1 search to find sensitive operator types."""
    strategy = cfg["strategy_stage1"]
    candidate_op_types = cfg["candidate_op_types"]
    target = cfg["target"]
    resume_enabled = cfg.get("resume", False)
    
    # Unpack state variables that will be modified
    ops_remaining = state["ops_remaining"]
    cumulative_exclusion_nodes = state["cumulative_exclusion_nodes"]
    per_op_results = state["per_op_results"]
    tipping_point_op_type = state["tipping_point_op_type"]
    tipping_point_score = state["tipping_point_score"]
    stage2_candidate_nodes = state["stage2_candidate_nodes"]

    pbar_s1 = tqdm(ops_remaining, desc=f"Stage 1/{part_name}", unit="op", total=len(candidate_op_types))
    
    # Common logic for handling a benchmark result and checking for a tipping point
    def check_for_tipping_point(op_type, current_score, nodes, last_score=None):
        fp32_level_met = (current_score <= reference_score) if minimize_metric else (current_score >= reference_score)
        if fp32_level_met:
            logging.info(f"✅ Early stop! FP32 performance level reached ({primary_metric.upper()}: {current_score:.4f} vs FP32 Ref: {reference_score:.4f}).")
            return True, op_type, current_score, list(nodes)

        # 'first' and 'percent' strategies look for the first significant improvement
        if strategy in ['first', 'percent']:
            is_improvement = (current_score < last_score) if minimize_metric else (current_score > last_score)
            if is_improvement:
                logging.info(f"✅ Tipping point! '{op_type}' improved {primary_metric.upper()} from {last_score:.4f} to {current_score:.4f}.")
                return True, op_type, current_score, list(nodes)
        
        # 'percent' strategy also checks if the target performance level is met
        if strategy == 'percent':
            performance_gap = abs(reference_score - baseline_score)
            improvement_needed = performance_gap * target
            target_score = (baseline_score - improvement_needed) if minimize_metric else (baseline_score + improvement_needed)
            target_met = (current_score <= target_score) if minimize_metric else (current_score >= target_score)
            if target_met:
                logging.info(f"✅ Target met! '{op_type}' reached the target score of {target_score:.4f}.")
                return True, op_type, current_score, list(nodes)

        return False, None, None, None

    # --- Strategy Implementations ---
    last_score = baseline_score
    best_overall_score = baseline_score
    best_op_type, best_exclusion_list_for_op = None, []

    for op_type in list(ops_remaining):
        nodes_to_add = nodes_by_type.get(op_type, [])
        if not nodes_to_add:
            logging.info(f"  ℹ️  No nodes of type '{op_type}' found in {part_name.upper()}; skipping.")
            per_op_results.append({"op_type": op_type, "node_count": 0, "score": None, "metrics": None})
            ops_remaining.remove(op_type)
            if resume_enabled: persist_state(stage=1, ops_left=ops_remaining, cumulative_nodes=cumulative_exclusion_nodes)
            continue
        
        pbar_s1.set_description(f"Stage 1/{part_name} (Testing {op_type}, {len(nodes_to_add)} nodes)")
        
        if strategy == 'best':
            current_test_exclusion_list = nodes_to_add
        else: # 'first' or 'percent'
            current_test_exclusion_list = cumulative_exclusion_nodes + nodes_to_add

        try:
            current_metrics, elapsed = quant_and_benchmark(current_test_exclusion_list)
        except Exception:
            logging.warning(f"Quant/benchmark failed for op type {op_type}. Skipping.", exc_info=True)
            current_metrics, elapsed = {primary_metric: baseline_score}, 0.0
        
        current_score = current_metrics[primary_metric]
        per_op_results.append({"op_type": op_type, "node_count": len(nodes_to_add), "score": current_score, "metrics": current_metrics, "time": elapsed})
        logging.info(f"  Result for '{op_type}': {primary_metric.upper()}={current_score:.4f} ({len(nodes_to_add)} nodes)")
        pbar_s1.set_postfix({primary_metric: f"{current_score:.4f}"})
        
        ops_remaining.remove(op_type)
        if strategy in ['first', 'percent']:
            cumulative_exclusion_nodes.extend(nodes_to_add)

        if resume_enabled:
            persist_state(stage=1, ops_left=ops_remaining, cumulative_nodes=cumulative_exclusion_nodes)

        # Check for early exit conditions
        found, tp_op, tp_score, s2_nodes = check_for_tipping_point(op_type, current_score, current_test_exclusion_list, last_score)
        if found:
            tipping_point_op_type, tipping_point_score, stage2_candidate_nodes = tp_op, tp_score, s2_nodes
            if resume_enabled: persist_state(stage=2, stage2_nodes=s2_nodes, best_list=s2_nodes, best_score=tp_score)
            break
        
        # Update state for next iteration
        last_score = current_score
        if strategy == 'best':
            is_best_so_far = (current_score < best_overall_score) if minimize_metric else (current_score > best_overall_score)
            if is_best_so_far:
                best_overall_score, best_op_type, best_exclusion_list_for_op = current_score, op_type, nodes_to_add
                logging.info(f"  Found new best score by excluding '{op_type}'.")
    
    pbar_s1.close()

    # Finalize 'best' strategy if no early stop occurred
    if strategy == 'best' and not tipping_point_op_type and best_op_type:
        logging.info(f"✅ Best tipping point found by excluding op type '{best_op_type}'.")
        tipping_point_op_type, tipping_point_score, stage2_candidate_nodes = best_op_type, best_overall_score, best_exclusion_list_for_op
        if resume_enabled: persist_state(stage=2, stage2_nodes=stage2_candidate_nodes, best_list=stage2_candidate_nodes, best_score=best_overall_score)

    return {
        "tipping_point_op_type": tipping_point_op_type,
        "tipping_point_score": tipping_point_score,
        "stage2_candidate_nodes": stage2_candidate_nodes,
        "cumulative_exclusion_nodes": cumulative_exclusion_nodes
    }

def _run_stage2_pruning(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    part_name: str,
    minimize_metric: bool,
    primary_metric: str,
    quant_and_benchmark: callable,
    persist_state: callable
) -> Tuple[List[str], float]:
    """Executes the Stage 2 pruning process to minimize the exclusion list."""
    strategy = cfg["strategy_stage2"]
    max_nodes_to_exclude = cfg["max_nodes_to_exclude"]
    resume_enabled = cfg.get("resume", False)
    
    # Unpack state
    stage2_candidate_nodes = state["stage2_candidate_nodes"]
    best_exclusion_list = state["best_exclusion_list"] or list(stage2_candidate_nodes)
    current_best_score = state["current_best_score"]
    nodes_left_for_pass = state["stage2_nodes_left"]
    
    logging.info(f"--- STAGE 2: Pruning nodes from {len(stage2_candidate_nodes)} candidates for {part_name.upper()} (Strategy: {strategy}) ---")
    if max_nodes_to_exclude is not None:
        logging.info(f"Will stop pruning if the exclusion list has {max_nodes_to_exclude} or fewer nodes.")
    logging.info(f"Initial {primary_metric.upper()} with all nodes: {current_best_score:.4f}")

    changed_in_pass = True
    pass_num = 0
    while changed_in_pass and (max_nodes_to_exclude is None or len(best_exclusion_list) > max_nodes_to_exclude):
        pass_num += 1
        changed_in_pass = False
        nodes_to_test_this_pass = nodes_left_for_pass or list(best_exclusion_list)
        if not nodes_to_test_this_pass: break
        
        logging.info(f"--- Pruning Pass #{pass_num} ({len(nodes_to_test_this_pass)} nodes left) ---")
        pbar_s2 = tqdm(nodes_to_test_this_pass, desc=f"Stage 2/{part_name}", unit="node", leave=False)

        for node in list(nodes_to_test_this_pass):
            pbar_s2.set_description(f"Stage 2/{part_name} (Test remove: {node.split('/')[-1]})")
            
            # Create a temporary list for testing
            temp_exclusion_list = [n for n in best_exclusion_list if n != node]
            
            try:
                metrics, _ = quant_and_benchmark(temp_exclusion_list)
                score = metrics[primary_metric]
                logging.info(f"  Test removing node '{node}' -> {primary_metric.upper()}: {score:.4f}")

                eps = 1e-6
                can_be_removed = False
                if strategy == 'relaxed' and ( (minimize_metric and score <= current_best_score + eps) or (not minimize_metric and score >= current_best_score - eps) ):
                    can_be_removed = True
                elif strategy == 'strict' and ( (minimize_metric and score < current_best_score - eps) or (not minimize_metric and score > current_best_score - eps) ):
                    can_be_removed = True
                
                if can_be_removed:
                    logging.info(f"✅ Node '{node}' removed. New {primary_metric.upper()}: {score:.4f}")
                    best_exclusion_list = temp_exclusion_list
                    current_best_score = score
                    changed_in_pass = True
                    # A node was removed, so the next pass will start with a fresh full list
                    nodes_left_for_pass = [] 
                    if resume_enabled: persist_state(stage=2, best_list=best_exclusion_list, best_score=current_best_score, nodes_left_for_stage2=nodes_left_for_pass)
                    break # Exit the for-loop to start a new pass
                else:
                    logging.info(f"❌ Node '{node}' cannot be removed without worsening the metric; re-adding it.")
            
            except Exception:
                logging.warning(f"Quant/benchmark failed when testing removal of node {node}. Re-adding node.", exc_info=True)
            
            # This node was tested and not removed, update the list of nodes left for this pass
            nodes_left_for_pass = [n for n in nodes_to_test_this_pass if n != node]
            if resume_enabled: persist_state(stage=2, best_list=best_exclusion_list, best_score=current_best_score, nodes_left_for_stage2=nodes_left_for_pass)
        
        pbar_s2.close()

    if max_nodes_to_exclude is not None and len(best_exclusion_list) > max_nodes_to_exclude:
         logging.warning(f"Pruning finished, but the exclusion list ({len(best_exclusion_list)} nodes) is larger than the max ({max_nodes_to_exclude}).")

    return best_exclusion_list, current_best_score

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
    quant_type: Any,
    quant_extra_options: Dict[str, Any],
    minimize_metric: bool,
    primary_metric: str,
    quant_test_dir: Path,
) -> Tuple[List[str], float]:
    """
    Runs the two-stage search process for a given model component (encoder or decoder).
    """
    state_path = quant_test_dir / f"{part_name}_search_state.json"
    resume_enabled = cfg.get("resume", False)

    # 1. Initialize or load the search state
    state = _initialize_search_state(
        state_path, part_name, cfg["candidate_op_types"], baseline_score, resume_enabled
    )

    # 2. Define helper closures that capture the context of this run
    def persist_state(stage: int, **kwargs):
        """Helper to save the current state to a file."""
        # Update the main state dict with any new values from kwargs
        for key, value in kwargs.items():
            if value is not None:
                state[key] = value
        
        data_to_save = {
            "part": part_name,
            "stage": stage,
            "ops_remaining": state["ops_remaining"],
            "cumulative_exclusion_nodes": state["cumulative_exclusion_nodes"],
            "per_op_results": state["per_op_results"],
            "tipping_point_op_type": state["tipping_point_op_type"],
            "tipping_point_score": state["tipping_point_score"],
            "stage2_candidate_nodes": state["stage2_candidate_nodes"],
            "best_exclusion_list": state["best_exclusion_list"],
            "current_best_score": state["current_best_score"],
            "stage2_nodes_left": kwargs.get("nodes_left_for_stage2", state["stage2_nodes_left"]),
        }
        try:
            _save_state_atomic(state_path, data_to_save)
        except Exception as e:
            logging.warning(f"Failed to persist search state to {state_path}: {e}")

    def quant_and_benchmark(exclusion_list: List[str]) -> Tuple[Dict[str, float], float]:
        """A wrapper to abstract away the temp file management and benchmarking call."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", dir=quant_test_dir, delete=False) as temp_model_file:
            temp_model_path = Path(temp_model_file.name)
        
        try:
            dynamic_quantization(
                model_to_search_path=model_to_search_path,
                out_path=temp_model_path,
                quant_type=quant_type,
                nodes_to_exclude=exclusion_list,
                extra_options=quant_extra_options
            )
            encoder_path = temp_model_path if part_name == "encoder" else fixed_encoder_path
            decoder_path = temp_model_path if part_name == "decoder" else fixed_decoder_path
            return run_benchmark(encoder_path, decoder_path, **run_benchmark_args)
        finally:
            if temp_model_path.exists():
                os.unlink(temp_model_path)

    # 3. Run Stage 1: Sensitivity Search
    if not state["stage1_skipped_due_to_resume"]:
        logging.info(f"--- STAGE 1: Discovering Sensitive Operation Types for {part_name.upper()} (Strategy: {cfg['strategy_stage1']}) ---")
        stage1_results = _run_stage1_search(
            state, cfg, part_name, nodes_by_type, baseline_score, reference_score,
            minimize_metric, primary_metric, quant_and_benchmark, persist_state
        )
        # Update state with results from stage 1
        state.update(stage1_results)

    # 4. Handle transition and check if Stage 1 was successful
    if not state["tipping_point_op_type"]:
        logging.error(f"Stage 1 for {part_name.upper()} failed: No operator type improved the baseline metric using strategy '{cfg['strategy_stage1']}'.")
        if resume_enabled: _remove_state(state_path)
        return [], baseline_score

    # Ensure stage2_candidate_nodes is initialized correctly from the state
    if not state["stage2_candidate_nodes"]:
        state["stage2_candidate_nodes"] = state["best_exclusion_list"] or state["cumulative_exclusion_nodes"]
    
    logging.info(f"\n✅ Stage 1 for {part_name.upper()} Complete. Tipping point: '{state['tipping_point_op_type']}'. Pool for Stage 2 has {len(state['stage2_candidate_nodes'])} nodes.\n")
    
    # 5. Run Stage 2: Pruning
    final_exclusion_list, final_score = _run_stage2_pruning(
        state, cfg, part_name, minimize_metric, primary_metric,
        quant_and_benchmark, persist_state
    )

    # 6. Finalization and Cleanup
    logging.info(f"\n--- {part_name.upper()} Pruning Complete ---")
    if not final_exclusion_list:
        logging.warning(f"Could not find any nodes to exclude from the {part_name.upper()} that improved accuracy.")
    else:
        logging.info(f"Found minimal exclusion list for {part_name.upper()} with {len(final_exclusion_list)} nodes, achieving a {primary_metric.upper()} of {final_score:.4f}")

    if resume_enabled:
        _remove_state(state_path)

    return final_exclusion_list, final_score