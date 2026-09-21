# LLM_GPU v0.4 Release Notes

## Summary

v0.4 completes the long-run GPU training validation of the homemade LLM on a physical NVIDIA CUDA GPU.

## Highlights

- Physical CUDA GPU training with PyTorch
- End-to-end training and inference validated on NVIDIA GeForce RTX 3070 Ti
- 754,560 trainable parameters
- Character-level Japanese tokenizer with vocabulary size 5,117
- 2-layer Transformer with causal single-head self-attention
- AdamW optimization and PyTorch autograd
- Long-run training configuration targeting approximately 10 hours
- Memory-efficient on-demand corpus window sampling
- v0.4 checkpoint: `model/model-gpu-v0.4.pt`
- Inference updated to load the v0.4 checkpoint

## Measured result

- Checkpoint loss: 3.082054258169556
- Inference throughput: approximately 139 tokens/s on RTX 3070 Ti
- Generated Japanese text shows improved sentence structure compared with v0.3

## Status

GPU end-to-end communication and execution validation is complete:
corpus -> tokenizer -> embedding -> transformer -> loss -> backward -> AdamW -> checkpoint -> reload -> GPU inference.
