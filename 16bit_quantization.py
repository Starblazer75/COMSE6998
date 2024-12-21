import onnx
from onnxconverter_common import float16

# CHANGE PATHS HERE
model_fp32_path = 'checkpoints/depth_anything_v2_vits.onnx'
model_fp16_path = 'checkpoints/depth_anything_v2_vits.quant.onnx'
# CHANGE PATHS HERE

model_fp32 = onnx.load(model_fp32_path)

model_fp16 = float16.convert_float_to_float16(model_fp32)

onnx.save(model_fp16, model_fp16_path)

print(f"Model converted to FP16 and saved at {model_fp16_path}")
