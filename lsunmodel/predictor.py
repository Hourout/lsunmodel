from collections import Counter


import cv2
import numpy as np
import torch

from lsunmodel.datasets import sequence
from lsunmodel.trainer import core

torch.backends.cudnn.benchmark = True


class Predictor:
    def __init__(self, weight_path, if_gpu=True):
        self.model = core.LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101')
        self.model.freeze()
        self.if_gpu = if_gpu
        if self.if_gpu:
            self.model.cuda()

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:
        if self.if_gpu:
            _, outputs = self.model(image.unsqueeze(0).cuda())
        else:
            _, outputs = self.model(image.unsqueeze(0))
        return outputs.cpu() if self.if_gpu else outputs
    
#     def predict_video(self, path, image_size=320, device=0, output='./output/outputVideo.mp4'):
#         stream = sequence.VideoStream(image_size, path, device)
#         video_writer = None
#         n = 0
#         for image in stream:
#             if video_writer is None:
#                 video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 30, stream.origin_size)
#             frame = self.feed(image).numpy()
#             axis = self.predict_axis(frame, stream.origin_size)
#             try:
#                 frame = self.plot_line(stream.frame, axis)
#                 video_writer.write(frame)
#             except:
#                 video_writer.write(stream.frame)
# #             video_writer.write(stream.frame)
#             if n>120:
#                 break
#             n += 1
#         video_writer.release()
        
    def predict_image(self, path, image_size=320):
        images = sequence.ImageFolder(image_size, path)
        for image, shape, _ in images:
            pred = self.feed(image)
        return pred, image, shape
    
    def predict_axis(self, pred, shape):
        axis = [self.find_axis(pred, [i,j] ) for i in range(318) for j in range(318)]
        axis = [i for i in axis if i!=0]
        axis = [axis[0]]+[j for i,j in enumerate(axis[1:]) if abs(j[0][0]-axis[i][0][0])>10 or abs(j[0][1]-axis[i][0][1])>10]
        axis = [[int(i[0][1]*shape[0]/320), int(i[0][0]*shape[1]/320), i[1]] for i in axis]
        return axis
    
    def find_axis(self, label, loc):
        cnt = Counter(label[0, loc[0]:loc[0]+3, loc[1]:loc[1]+3].flatten())
        if len(cnt)==1:
            point = 0
        elif len(cnt)==2:
            if loc[0]==0:
                point = ([0,0], 0) if loc[1]==0 else ([0, loc[1]+1],0)
            elif loc[1]==0:
                point = ([320,0],0) if loc[0]==317 else ([loc[0]+1, 0],0)
            elif loc[0]==317:
                point = ([320, loc[1]+1],0)
            elif loc[1]==317:
                point = ([loc[0]+1, 320],0)
            else:
                point = 0
        else:
            point = ([loc[0]+1, loc[1]+1],1)
        return point
    
    def find_line(self, image, axis):
        edge = [i[:2] for i in axis if i[2]==0]
        center = [i[:2] for i in axis if i[2]==1]
        line = [sorted([i+j+[(i[0]-j[0])**2+(i[1]-j[1])**2] for j in center], key=lambda x:x[4])[0][:4] for i in edge]
        if len(center)!=1:
            r = 2 if len(center)==2 else 4
            line1 = sorted([i+j+[(i[0]-j[0])**2+(i[1]-j[1])**2] for k,i in enumerate(center) for j in center[k+1:]], key=lambda x:x[4])[:r]
            line += [i[:4] for i in line1]
        return line
    
    def plot_line(self, image, axis):
        line = self.find_line(image, axis)
        for i in line:
            cv2.line(image, (i[0],i[1]), (i[2],i[3]), (255,0,0), 2)
        return image
    
    def plot_segmentation(self, image, pred, alpha=0.4):
        label = core.label_as_rgb_visual(pred).squeeze(0)
        blend_output = (image / 2 + .5) * (1 - alpha) + (label * alpha)
        blend_output = blend_output.permute(1, 2, 0).numpy()
        return (blend_output[..., ::-1] * 255).astype(np.uint8)

    def predict_area_rate(self, path, threshold_max=0.5, threshold_min=0.15):
        pred, img, shape = self.predict_image(path, image_size=320)
        pred = pred.numpy()
        label = {0:'Frontal wall', 1:'Left wall', 2:'Right wall', 3:'Floor', 4:'Ceiling'}
        cnt = {label[i]:j/pred.size for i,j in Counter(pred.flatten()).items()}
        cnt = [i for i,j in cnt.items() if j>threshold_max or j<threshold_min]
        return '面积比例合格' if len(cnt)==0 else '面积比例不合格'
    
    def predict_vertical(self, line, threshold=0.06):
        pred = [i for i in line if abs(i[1]-i[3])>10*abs(i[0]-i[2]) and abs(i[0]-i[2])/abs(i[1]-i[3])>threshold]
        return '墙角线不垂直' if len(pred)>0 else '墙角线垂直'
        
# import time
# path = '../surface_relabel/val/008fd415c7696c568ab00453085c2f356f1dcf88.jpg'
# image = cv2.imread(path)
# predictor = Predictor(weight_path='../model_retrained4.ckpt')
# a = time.time()
# pred, img, shape = predictor.predict_image(path)
# axis = predictor.predict_axis(pred.numpy(), shape)
# image = predictor.plot_line(image, axis)
# la.image.array_to_image(image)
# image = predictor.plot_segmentation(img, pred, alpha=.4)
# la.image.array_to_image(image)
# print(time.time()-a)

