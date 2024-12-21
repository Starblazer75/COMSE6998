import argparse
import onnx
from onnxconverter_common import float16

def main():
    parser = argparse.ArgumentParser(description="Convert an ONNX model from FP32 to FP16.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input FP32 ONNX model.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the FP16 ONNX model.")
    args = parser.parse_args()

    model_fp32_path = args.input
    model_fp16_path = args.output

    print(f"Loading FP32 model from: {model_fp32_path}")
    model_fp32 = onnx.load(model_fp32_path)

    print("Converting model to FP16...")
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    print(f"Saving FP16 model to: {model_fp16_path}")
    onnx.save(model_fp16, model_fp16_path)
    print("Conversion to FP16 completed successfully.")

if __name__ == "__main__":
    main()
