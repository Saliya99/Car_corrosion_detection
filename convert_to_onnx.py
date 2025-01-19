import torch
import onnx

model = torch.load('best_model.pth')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  

torch.onnx.export(model, dummy_input, "best_model.onnx")
print("Model converted to ONNX format")
