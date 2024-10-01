import streamlit as st
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.enable_model_cpu_offload()

st.title("AI Image Generator") 



# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

#prompt = "An astronaut riding a green horse"
prompt = st.text_input("Enter Prompt")
if prompt:
    with st.spinner('Processing'):
        images = pipe(prompt=prompt).images[0]
    st.success("Done!")
    st.image(images, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
