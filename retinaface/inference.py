import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from retinaface.config import get_backbone_cfg, get_default_weight_path
from retinaface.model import RetinaFace
from retinaface.postprocess import Anchors, decode, decode_landm, non_max_suppression, retinaface_correct_boxes


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


class Retinaface:
    _defaults = {
        'model_path': '',
        'backbone': 'mobilenetv2_050',
        'confidence': 0.5,
        'nms_iou': 0.45,
        'input_shape': [1280, 1280, 3],
        'letterbox_image': True,
        'cuda': True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if not self.model_path:
            self.model_path = get_default_weight_path(self.backbone)

        self.cfg = get_backbone_cfg(self.backbone)
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()
        print(f'{self.model_path} model, and classes loaded.')
        if self.cuda:
            self.net = self.net.cuda()

    def detect_image(self, image):
        old_image = image.copy()
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
        ]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()

            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return old_image

            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        for box in boxes_conf_landms:
            text = '{:.4f}'.format(box[4])
            box = list(map(int, box))
            cv2.rectangle(old_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(old_image, text, (box[0], box[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.circle(old_image, (box[5], box[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (box[7], box[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (box[9], box[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (box[11], box[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (box[13], box[14]), 1, (255, 0, 0), 4)
        return old_image

    def get_fps(self, image, test_interval):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()
            self.net(image)

        start = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                loc, conf, landms = self.net(image)
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                conf = conf.data.squeeze(0)[:, 1:2]
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                non_max_suppression(boxes_conf_landms, self.confidence)
        return (time.time() - start) / test_interval

    def get_map_txt(self, image):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
        ]

        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()
            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms) <= 0:
                return np.array([])
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        return boxes_conf_landms


def run_predict(input_dir, output_dir, backbone='mobilenet', weights=''):
    detector = Retinaface(backbone=backbone, model_path=weights)
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(input_dir, img_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect_image(image)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, img_name), result)
    print(f'检测完成！结果保存到 {output_dir}')


def run_video(video_path, save_path, fps, backbone='mobilenet', weights=''):
    detector = Retinaface(backbone=backbone, model_path=weights)
    capture = cv2.VideoCapture(video_path)
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(save_path, fourcc, fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError('无法读取视频/摄像头')

    fps_counter = 0.0
    while True:
        start = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(detector.detect_image(frame))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps_counter = (fps_counter + (1.0 / (time.time() - start))) / 2
        frame = cv2.putText(frame, f'fps= {fps_counter:.2f}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('video', frame)
        if save_path:
            out.write(frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

    capture.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()
    print('视频检测完成！')


def run_fps(image_path, test_interval, backbone='mobilenet', weights=''):
    detector = Retinaface(backbone=backbone, model_path=weights)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tact_time = detector.get_fps(image, test_interval)
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')