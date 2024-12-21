import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# CHANGE PATHS HERE
model_fp32 = 'checkpoints/depth_anything_v2_vits.onnx'
model_quant = 'checkpoints/depth_anything_v2_vits.quant.onnx'
# CHANGE PATHS HERE

quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)