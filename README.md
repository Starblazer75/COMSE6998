## Pre-trained Models

There are three models provided by the Github (These are the same links as the github, so go to the DepthAnythingV2 Github if you don't trust these):

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |


## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Download the checkpoints listed and put them under the `checkpoints` directory.

### Scripts Used
Initial Inference with normal weights. Use --extension to add on the name after vits, vitl, etc.
```bash
python3 run.py --encoder vits --img-path DA-2K/images --outdir initial_weights --grayscale --extension pruned
```

ONNX Inference 
```bash
python3 run_onnx.py --img-path DA-2K/images --outdir onnx_weights --grayscale --onnx-model-path ../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.quant.onnx
```

ONNX 16 bit 
```bash
python3 run_16onnx.py --img-path DA-2K/images --outdir onnx_weights --grayscale --onnx-model-path ../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.quant.onnx
```

Prune
```bash
python3 prune.py --input-model checkpoints/depth_anything_v2_vits.pth --output-model checkpoints/depth_anything_v2_vits_pruned.pth --pruning-percentage 0.01
```

Change to ONNX Architecture
```bash
python3 export_onnx.py --input checkpoints/depth_anything_v2_vits.pth --output checkpoints/depth_anything_v2_vits.onnx
```

Quantize to FP16 ONNX Weight
```bash
python3 16bit_quantization.py --input checkpoints/depth_anything_v2_vits.onnx --output checkpoints/depth_anything_v2_vits.quant.onnx' 
```

Quantize to QUINT8 ONNX Weight
```bash
python3 8bit_quantization.py --input checkpoints/depth_anything_v2_vits.onnx --output checkpoints/depth_anything_v2_vits.quant.onnx'
```

Accuracy
```bash
python3 accuracy.py --annotations DA-2K/annotations.json --image-dir fine_weights
```

PyTorch Profiler
```bash
python3 bottleneck.py --encoder vits --img-path DA-2K/images --outdir initial_weights --grayscale
```

### Strategy Used for Fine-Tuning

Using the large model weight, I split half of the benchmark images into a training section and an inference section. Afterwards, I created ground-truth data for the training section with the large model weight so I can then fine-tune the small model weight with the training data and ground truth. 

```bash
python3 generate_groundtruth.py --encoder vitl --img-path DA-2K/images/ --outdir DA-2K/ --extension pruned
```

Now that you have the val directory of images containing the inference images and the depth directory containing the original images as well as their ground-truth values, you can fine-tune the model.

```bash
python3 fine_tune.py --input checkpoints/depth_anything_v2_vits_pruned.pth \
                     --output checkpoints/depth_anything_v2_vits_finetuned.pth \
                     --image-dir DA-2K/images \
                     --depth-dir DA-2K/depths \
                     --batch-size 8 \
                     --learning-rate 1e-4 \
                     --num-epochs 10
```


## DA-2K Evaluation Benchmark

Download DA-2K Benchmark Images Here: [DA-2K benchmark](./DA-2K.md). (Link is also in original Git)

## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation.

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
