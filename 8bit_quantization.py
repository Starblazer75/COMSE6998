import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model dynamically to QUInt8.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input FP32 ONNX model.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the quantized ONNX model.")
    args = parser.parse_args()

    model_fp32 = args.input
    model_quant = args.output

    print(f"Quantizing model:\nInput: {model_fp32}\nOutput: {model_quant}")
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print("Quantization completed successfully.")

if __name__ == "__main__":
    main()
