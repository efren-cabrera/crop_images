from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .AbstractCropper import AbstractCropper

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

classes = pkg_resources.read_text(__package__, 'yolov3-classes.txt').splitlines()
classes_colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolov3_weights = pkg_resources.path(__package__, 'yolov3.weights')
yolov3_cfg = pkg_resources.path(__package__, 'yolov3.cfg')
with yolov3_weights as weights, yolov3_cfg as cfg:
    net = cv2.dnn.readNet(weights.as_posix(), cfg.as_posix())

class YOLOCropper(AbstractCropper):
    def crop(self, image, relation: Tuple):
        height, width, _ = image.shape

        boxes, indices, _, _ = self.detect(image)
        centers = []
        xx = []
        yy = []
        z = []
        areas = []
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            center = [round((w/2) + x), round((h/2) + y)]
            centers.append(center)
            x_grid, y_grid, z_grid = self.get_values(center, width, height)
            xx.append(x_grid)
            yy.append(y_grid)
            z.append(z_grid)
            areas.append(w*h)

        z_sum = np.zeros_like(z[0])
        for area, z_grid in zip(areas, z):
            #z_sum = z_sum + z_grid
            z_sum = z_sum + (z_grid*area) # Add weight by area
        z_sum = (z_sum - np.min(z_sum))/np.ptp(z_sum)
        cs = plt.contour(xx[0], yy[0], z_sum, levels=[.3])
        plt.contourf(xx[0], yy[0], z_sum, alpha=.6)
        cn = cs.collections[0].get_paths()
        vs=[]
        for i in range(len(cn)):
            vs.append(cn[i].vertices)
        vs = np.array(vs[0])
        x_polygon = vs[:,0]
        y_polygon = vs[:,1]
        center_polygon = [x_polygon.mean(), y_polygon.mean()]

        crop_width, crop_height = self.find_crop_width_height(width, height, relation)

        x_0 = center_polygon[0] - (crop_width/2)
        x_f = center_polygon[0] + (crop_width/2)

        x_0, x_f = self.adjust_values(x_0, x_f, 0, width)

        y_0 = center_polygon[1] - (crop_height/2)
        y_f = center_polygon[1] + (crop_height/2)

        y_0, y_f = self.adjust_values(y_0, y_f, 0, height)
        return self.make_crop(image, x_0, x_f, y_0, y_f)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def get_values(self, center, widht, height):
        x = np.linspace(-widht, widht, widht*2)
        y = np.linspace(-height, height, height*2)
        xx, yy = np.meshgrid(x, y)
        z = xx**2 + yy**2
        xx = xx + center[0]
        yy = yy + center[1]
        xx = xx[height-center[1]:height*2-center[1], widht-center[0]:widht*2-center[0]]
        yy = yy[height-center[1]:height*2-center[1], widht-center[0]:widht*2-center[0]]
        z = z[height-center[1]:height*2-center[1], widht-center[0]:widht*2-center[0]]
        z = (z - np.min(z))/np.ptp(z)
        return xx, yy, z

    def detect(self, image):
        height, width, _ = image.shape
        scale = 0.00392


        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(self.get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return boxes, indices, confidences, class_ids
    