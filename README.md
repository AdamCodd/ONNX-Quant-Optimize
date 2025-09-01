# ONNX-Quant-Optimize: Optimizing ONNX Quantization

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

I made this tool because the default dynamic ONNX quantization, while fast, often degrades the performance of finetuned models, especially those with subgraphs. Excluding a few nodes restores performance instantly without keeping everything in FP32 and sacrificing  inference speed. To do that, this script employs a two-stage workflow to identify the optimal set of nodes to exclude from quantization in ONNX encoder-decoder models.

## Key Features

*   **Two-Stage Optimization Workflow**: A systematic approach to first identify sensitive operation types and then prune the exclusion list to a minimal set.
*   **Broad Compatibility**: Designed for encoder-decoder models and supports both encoder, decoder, or combined searches.
*   **Multiple Search Strategies**: Offers various strategies (`first`, `best`, `percent`) to discover the initial set of sensitive nodes, providing flexibility for different optimization needs.
*   **Performance Benchmarking**: Integrated benchmarking to evaluate model performance using a suite of metrics (WER, CER, BLEU, ROUGE) and compare it against FP32 and fully quantized baselines.
*   **Automated Model Export**: The script can automatically export the final, optimally quantized models for immediate use.

## How It Works

The script automates the process of discovering which nodes in an ONNX model are most sensitive to quantization. It does this by:

1.  **Establishing Baselines**: It first benchmarks the original FP32 model and a fully quantized version to establish performance boundaries.
2.  **Stage 1: Sensitive Operator Discovery**: The script iteratively excludes nodes by their operator type (e.g., `MatMul`, `Add`, `Gelu`) and measures the impact on a primary performance metric. This stage identifies a "tipping point" operator, where excluding it and all preceding operator types brings the model's performance to a desired level.
3.  **Stage 2: Pruning**: Starting with the broad list of nodes from Stage 1, this stage systematically re-includes nodes one by one to find the smallest possible set of excluded nodes that maintains the performance gains.
4.  **Final Evaluation**: The script provides a final benchmark of the partially quantized model and a detailed comparison against the baselines.

## Getting Started

### Prerequisites

*   Python 3.8+
*   ONNX Runtime
*   Transformers
*   Optimum
*   Additional libraries for metrics: `jiwer` for WER/CER and `evaluate` for BLEU/ROUGE.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AdamCodd/ONNX-Quant-Optimize.git
    cd ONNX-Quant-Optimize
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements-cpu.txt
    ```
or if you're testing on GPU:
    ```bash
    pip install -r requirements-gpu.txt
    ```

### Usage

1.  **Configure your settings:**
    Create a `config_quant.json` file to specify model paths, search strategies, and other parameters. A documented example of this file is provided in the repo.

2.  **Provide evaluation samples:**
    Create a `samples.jsonl` file with prompts and ground truth references for benchmarking. Each line should be a JSON object:
    ```json
    {"input": "Your input prompt here.", "ground_truth": "The expected output."}
    ```

3.  **Run the script:**
    ```bash
    python discover_quant_exclusion.py --config config_quant.json
    ```
    Export the quantized ONNX encoder/decoder (optionally):
     ```bash
    python discover_quant_exclusion.py --config config_quant.json --export_final path/to/your/directory
    ```   

## Configuration

The script is controlled by a `config_quant.json` file. Here are some of the key options:

*   `"model_dir"`: Path to the directory containing the source model (e.g., in Hugging Face format).
*   `"onnx_dir"`: Directory to save the exported ONNX models.
*   `"quant_test_dir"`: A temporary directory for intermediate quantized models.
*   `"search_target"`: The part of the model to search. Options: `"encoder"`, `"decoder"`, `"both"`.
*   `"metrics"`: A list of metrics to evaluate. Options: `"wer"`, `"cer"`, `"bleu"`, `"rouge"`.
*   `"primary_metric"`: The main metric to use for optimization decisions.
*   `"strategy_stage1"`: The strategy for the first stage of the search. Options:
    *   `"first"`: Stops at the first operator type that improves the score.
    *   `"best"`: Tries all operator types and picks the one with the best cumulative score.
    *   `"percent"`: Aims to recover a certain percentage of the performance gap between the fully quantized and FP32 models.
*   `"strategy_stage2"`: The strategy for the pruning stage. Options:
    *   `"relaxed"`: A node is kept excluded if removing it doesn't degrade performance below the "tipping point" score from Stage 1.
    *   `"strict"`: A node is kept excluded only if removing it degrades the *current best* score.

## Example Output

After running, the script will output the recommended exclusion lists for the encoder and decoder, along with a summary table comparing the performance of the different model versions:

```
========================= FINAL RESULTS =========================
Found minimal exclusion list for DECODER with 8 nodes.
RECOMMENDATION: Use this list for the 'nodes_to_exclude' argument when quantizing the decoder.
--------------------------------------------------------------------------------
DECODER_NODES_TO_EXCLUDE = [
    '/model/decoder/layers/0/encoder_attn/MatMul',
    '/model/decoder/layers/1/encoder_attn/MatMul',
    ...
]
--------------------------------------------------------------------------------

No beneficial nodes to exclude were found for the ENCODER.


📊============== FINAL SUMMARY ==============📊
Model                                    Time (s)               ΔTime      WER         CER      ΔWER        ΔCER        Filesize
---------------------------------------------------------------------------------------------------------------------------------
FP32 Reference                          15.2345                  -     0.1234      0.0456        -           -          950.23MB
QUInt8 Dynamic (Fully Quantized)        8.1234    -7.111s (x0.53)      0.1876      0.0789     +0.0642     +0.0333       240.12MB
QUInt8 Dynamic (Partially Quantized)    8.5678    -6.667s (x0.56)      0.1255      0.0467     +0.0021     +0.0011       245.34MB
```
