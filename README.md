# GGUF Model Quantization 🚀

A lightweight tool for quantizing large language models to GGUF format with configurable bit precision.

## Features ✨

- Support for 4-bit and 8-bit quantization
- Compatible with Hugging Face models
- Memory efficient processing
- Simple API for custom implementations
- Built-in scaling factor calculation
- Automatic tensor type handling

## Installation 🛠️

```bash
git clone https://github.com/KevinDKao/gguf-quantization
cd gguf-quantization
pip install -r requirements.txt
```

## Dependencies 📦

- torch
- transformers
- numpy
- gguf

## Quick Start 🏃‍♂️

```python
from quantize import quantize_model

model_path = "path/to/model"
output_path = "quantized_model.gguf"

# Quantize to 4-bit
quantize_model(model_path, output_path, bits=4)
```

## Advanced Usage 🔧

```python
# 8-bit quantization
quantize_model("gpt2", "gpt2_quantized.gguf", bits=8)

# Custom model quantization
model = AutoModelForCausalLM.from_pretrained("custom_model")
tokenizer = AutoTokenizer.from_pretrained("custom_model")
quantize_model("custom_model", "custom_quantized.gguf", bits=4)
```

## How It Works 🤔

1. Loads model and tokenizer from specified path
2. Calculates optimal scaling factors for quantization
3. Converts float32 tensors to int4/int8 with scaling
4. Preserves non-float tensors in original format
5. Writes quantized model to GGUF format
6. Automatically handles tokenizer configuration

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

MIT License

## Contact 📬

- GitHub: [@KevinDKao](https://github.com/KevinDKao)
- Issues: [https://github.com/KevinDKao/gguf-quantization/issues](https://github.com/KevinDKao/gguf-quantization/issues)

## Star History 🌟

[![Star History Chart](https://api.star-history.com/svg?repos=KevinDKao/gguf-quantization&type=Date)](https://star-history.com/#KevinDKao/gguf-quantization&Date)