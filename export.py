from omdet.inference.det_engine import DetEngine
import torch

if __name__ == "__main__":
    model_dir = "./resources"
    batch_size = 16  # 可根据需要调整批量大小
    img_tensor =  torch.rand(batch_size, 3, 640, 640)  # 调整批量大小
    label_feats = torch.rand(80, batch_size, 512)  # 调整批量大小，80 is cls num, 512 is clip dim
    task_feats = torch.rand(77, batch_size, 512)  # 调整批量大小，77 is task dim
    task_mask = torch.rand(batch_size, 77)  # 调整批量大小

    engine = DetEngine(model_dir=model_dir, batch_size=batch_size, device='cpu')
    onnx_model_path = "./omdet.onnx"
    engine.export_onnx('OmDet-Turbo_tiny_SWIN_T', img_tensor, label_feats, task_feats, task_mask, onnx_model_path)