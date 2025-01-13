import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from gguf import GGUFWriter, Tensor, TensorType

def quantize_model(model_path, output_path, bits=4):
   # Load model and tokenizer
   model = AutoModelForCausalLM.from_pretrained(model_path)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   
   # Get model weights
   state_dict = model.state_dict()
   
   # Create GGUF writer
   writer = GGUFWriter(output_path, Tensor)
   
   # Add tokenizer config
   writer.add_tokenizer_config(tokenizer)
   
   # Quantize and add tensors
   for name, param in state_dict.items():
       tensor = param.detach().cpu().numpy()
       
       if tensor.dtype == np.float32:
           # Calculate scaling factor
           scale = (2**(bits-1) - 1) / np.max(np.abs(tensor))
           
           # Quantize to int8/int4
           quantized = np.clip(np.round(tensor * scale), -2**(bits-1), 2**(bits-1)-1)
           quantized = quantized.astype(np.int8 if bits==8 else np.int4)
           
           # Add tensor
           writer.add_tensor(
               name,
               quantized,
               TensorType.INT8 if bits==8 else TensorType.INT4,
               scale
           )
       else:
           writer.add_tensor(name, tensor)
   
   writer.write_header()
   writer.close()

# Usage
model_path = "path/to/model"
output_path = "quantized_model.gguf" 
quantize_model(model_path, output_path, bits=4)