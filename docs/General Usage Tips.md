# CPU Inference

Huggingface has a dedicated page: https://huggingface.co/docs/transformers/perf_infer_cpu

GGUF is supported in `transformers` as of v4.41: https://github.com/huggingface/transformers/releases/tag/v4.41.0

# Model Instantiation dtype

The size of a model is determined by the number of its parameters, and their precision, typically one of float32, float16 or bfloat16. 

Using a lower precision dtype like float16 or bfloat16 reduces the memory footprint of the model.

Many modern GPUs and TPUs have specialized hardware designed to accelerate mixed-precision computations, especially those involving float16 or bfloat16 types. These computations can be significantly faster than their float32 counterparts, leading to quicker training times and faster inference. NVIDIA's latest GPUs have tensor cores optimized for float16 calculations, while Google's TPUs are optimized for bfloat16.

Almost all models are trained in bfloat16 nowadays, there is no reason to run the model in full float32 precision if your GPU supports bfloat16. Float32 won’t give better inference results than the precision that was used to train the model.

If you are unsure in which format the model weights are stored on the Hub, you can always look into the checkpoint’s config under "torch_dtype". It is recommended to set the model to the same precision type as written in the config when loading with from_pretrained(..., torch_dtype=...) except when the original type is float32 in which case one can use both float16 or bfloat16 for inference.

Under Pytorch a model normally gets instantiated with torch.float32 format. To overcome this limitation, you can either explicitly pass the desired dtype using torch_dtype argument:

`model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)`

Or, if you want the model to always load in the most optimal memory pattern, you can use the special value "auto", and then dtype will be automatically derived from the model’s weights:

`model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")`

References:
* https://huggingface.co/docs/transformers/en/main_classes/model