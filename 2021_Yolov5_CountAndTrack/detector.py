import json
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

class Get_Pred():
    def __init__(self, modelpath):
        self.modelpath = modelpath
        self.imgsz = 640
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self, img):
        model = attempt_load(self.modelpath, map_location=self.device)
        names = model.module.names if hasattr(model, 'module') else model.names
        num = 0
        showimg = img.copy()
        pred_list = []
        with torch.no_grad():
            img = letterbox(img, new_shape=640)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 前向推理
            pred = model(img, augment=False)[0]

            # NMS去除多余的框
            pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
            # print(pred)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        num = num + 1
                        str_xyxy = (torch.tensor(xyxy).tolist())
                        pred_list.append(str_xyxy)

        pred_dict = {
                    'total num': num,
                    'xyxy': pred_list,
                }

        pred_json = json.dumps(pred_dict, sort_keys=True, indent=4, separators=(',', ':'))
        return pred_json

