# Use the base image of PyTorch with CUDA 12.1 and cuDNN 9
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Install dependencies for LLaMA support
RUN pip install transformers==4.40.0 accelerate

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# Make a directory for the model
RUN mkdir /ckpt

# Download the model from Hugging Face
RUN huggingface-cli download MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M --local-dir=/ckpt

# Add a default command
CMD ["/bin/bash"]
