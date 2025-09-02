# quant_utils.py

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import onnx

class SuppressQuantFilter(logging.Filter):
    """A logging filter to suppress specific verbose messages from ONNX Runtime quantization."""
    def filter(self, record):
        msg = record.getMessage().lower()
        patterns = [
            "quantization parameters for tensor",
            "ignore matmul due to non constant b",
            "please consider to run pre-processing before quantization.",
            "inference failed or unsupported type to quantize for tensor",
            "tensor_type {"
        ]
        for p in patterns:
            if p in msg:
                return False
        return True

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

def ensure_named_nodes(onnx_model: onnx.ModelProto, min_named_fraction: float = 0.01) -> bool:
    """
    Check if ONNX model node names are present. Returns True if enough nodes are named.
    If a model has *very few* named nodes, node-based exclusion search will likely fail.
    """
    nodes = onnx_model.graph.node
    if not nodes:
        return False
    named = sum(1 for n in nodes if n.name)
    frac = named / max(1, len(nodes))
    logging.debug(f"Named nodes: {named}/{len(nodes)} ({frac:.2%})")
    return frac >= min_named_fraction

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

def _save_state_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON state atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(str(tmp), str(path))
        logging.debug(f"Saved state to {path}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def _load_state(path: Path) -> Optional[Dict[str, Any]]:
    """Loads the search state from a JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.warning(f"Failed to load resume state {path}: {e}")
        return None

def _remove_state(path: Path) -> None:
    """Removes the state file."""
    try:
        if path.exists(): path.unlink()
    except Exception:
        pass

def _cleanup_stage1_temp_dirs(quant_test_dir: Path) -> None:
    """
    Remove leftover temporary directories created during Stage 1 that do not match
    the per-process worker dir naming (worker_{pid}). This helps free disk space
    before starting Stage 2. Non-fatal on errors (logs warnings).
    """
    if not quant_test_dir.exists():
        return
    for item in quant_test_dir.iterdir():
        try:
            if item.is_dir() and not item.name.startswith("worker_"):
                # remove directory and all contents
                shutil.rmtree(item)
                logging.info(f"Removed Stage 1 temp dir: {item}")
        except Exception as e:
            logging.warning(f"Failed to remove temp directory {item}: {e}")
