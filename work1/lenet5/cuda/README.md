# Naive CUDA LeNet-5

`naive/` contains exactly one CUDA source file for each logical stage:

```text
conv1 → pool1 → conv2 → pool2 → conv3 → fc1 → fc2
```

There are no binary weight files and no parameter copies stored in extensions. `CudaLeNet5` has the same `state_dict` layout as `model.LeNet5`; use normal PyTorch loading, then call `load_kernels()`:

```python
reference = LeNet5()
reference.load_state_dict(torch.load("checkpoint.pt", weights_only=True))

cuda_model = CudaLeNet5().cuda()
cuda_model.load_state_dict(reference.state_dict())
cuda_model.load_kernels()
logits = cuda_model(images.cuda())
```

Check the naive path against PyTorch:

```bash
PATH=/data/workspace/haoyu/software/miniconda3/envs/kernel/bin:$PATH \
  python model_cuda.py --batch 16
```
