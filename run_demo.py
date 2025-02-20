import os
import time
from omdet.inference.det_engine import DetEngine
from omdet.utils.plots import Annotator
from PIL import Image
import numpy as np
from safetensors.torch import load_file
import torch

# Check if the converted PyTorch checkpoint exists; if not, convert from safetensors.
model_path = "resources/OmDet-Turbo_tiny_SWIN_T.pth"
safetensor_path = "resources/model.safetensors"
if not os.path.exists(model_path):
    print(f"{model_path} not found. Converting from safetensors...")
    ckpt = load_file(safetensor_path)
    torch.save(ckpt, model_path)
    print("Conversion complete.")

if __name__ == "__main__":
    engine = DetEngine(batch_size=128, device='cuda')
    img_paths = ['./sample_data/000000574769.jpg', './sample_data/161030_nns_machette_16x9_992.jpg'] * 200  # 400 images total
    labels = ["person", "cat", "orange"]  # Labels to be predicted
    prompt = 'Detect {}.'.format(','.join(labels))  # Task prompt

    # Start timing the inference for benchmarking average throughput per second (number of frames)
    start_time = time.perf_counter()
    res = engine.inf_predict(
        'OmDet-Turbo_tiny_SWIN_T',  # prefix name of the pretrained checkpoints
        task=prompt,
        data=img_paths,
        labels=labels,
        src_type='local',           # type of the image paths, "local"/"url"
        conf_threshold=0.30,
        nms_threshold=0.5
    )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    throughput = len(img_paths) / total_time
    print(f"Inference took {total_time:.2f} seconds for {len(img_paths)} images. Throughput: {throughput:.2f} images/second.")

    out_folder = './outputs'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Process each image and save the annotated outputs
    for idx, img_path in enumerate(img_paths):
        im = Image.open(img_path)
        a = Annotator(
            np.ascontiguousarray(im),
            font_size=12,
            line_width=1,
            pil=True,
            font='sample_data/simsun.ttc'
        )
        for R in res[idx]:
            a.box_label(
                [R['xmin'], R['ymin'], R['xmax'], R['ymax']],
                label=f"{R['label']} {str(int(R['conf'] * 100))}%",
                color='red'
            )
        image = a.result()
        img = Image.fromarray(image)
        img.save(os.path.join(out_folder, os.path.basename(img_path)))