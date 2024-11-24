# Copyright 2024 Adobe. All rights reserved.

from run_magicfu import MagicFixup
import os
import pathlib
import torchvision
from torch import autocast
from PIL import Image
import gradio as gr
import numpy as np
import argparse
import torch
import time
import tempfile

def sample(original_image, coarse_edit, step_count, num_samples):
    to_tensor = torchvision.transforms.ToTensor()
    outputs = []
    with autocast("cuda"):
        w, h = coarse_edit.size
        # Bemeneti képek előkészítése
        ref_image_t = to_tensor(original_image.resize((512,512), Image.Resampling.LANCZOS)).half().cuda()
        coarse_edit_t = to_tensor(coarse_edit.resize((512,512), Image.Resampling.LANCZOS)).half().cuda()
        coarse_edit_mask_t = to_tensor(coarse_edit.resize((512,512), Image.Resampling.LANCZOS)).half().cuda()
        mask_t = (coarse_edit_mask_t[-1][None, None,...]).half()
        coarse_edit_t_rgb = coarse_edit_t[:-1]
        
        for _ in range(num_samples):
            # Modell futtatása
            out_rgb = magic_fixup.edit_image(ref_image_t, coarse_edit_t_rgb, mask_t, start_step=1.0, steps=step_count)
            
            # Debug információk
            print(f"Model output shape: {out_rgb.shape}")
            print(f"Model output min/max: {out_rgb.min().item():.4f}/{out_rgb.max().item():.4f}")
            
            # Alapvető képfeldolgozás
            output = out_rgb.squeeze().cpu().detach().float()
            
            # Közvetlen konverzió [0,1] tartományból [0,255] tartományba
            output = (output * 255).clamp(0, 255).to(torch.uint8)
            output = output.moveaxis(0, -1).numpy()
            
            # PIL kép létrehozása
            output_pil = Image.fromarray(output, mode='RGB')
            if w != 512 or h != 512:
                output_pil = output_pil.resize((w, h), Image.Resampling.LANCZOS)
            
            outputs.append(output_pil)
        
        return outputs

def file_exists(path):
    """ Check if a file exists and is not a directory. """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return path

def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(description="Process images based on provided paths.")
    parser.add_argument("--checkpoint", type=file_exists, required=True, help="Path to the MagicFixup checkpoint file.")

    return parser.parse_args()

demo = gr.Interface(
    fn=sample,
    inputs=[
        gr.Image(type="pil", image_mode='RGB', label="Eredeti kép"),
        gr.Image(type="pil", image_mode='RGBA', label="Szerkesztett kép"),
        gr.Slider(minimum=20, maximum=100, value=50, step=1, label="Lépések száma"),
        gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Generált képek száma")
    ],
    outputs=gr.Gallery(label="Eredmények", columns=2, rows=2),
    examples='examples',
    title="MagicFixup Képszerkesztő",
    description="Objektumok mozgatása képeken magas minőségben"
)
    
if __name__ == "__main__":
    args = parse_arguments()

    # create magic fixup model
    magic_fixup = MagicFixup(model_path=args.checkpoint)
    demo.launch(share=True)   
