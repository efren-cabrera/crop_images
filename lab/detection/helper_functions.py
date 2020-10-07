import numpy as np
import cv2

with open("./yolov3-classes.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
classes_colors = np.random.uniform(0, 255, size=(len(classes), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = classes_colors[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_values(center, widht, height):
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
