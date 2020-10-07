import numpy as np
import cv2
import matplotlib.pyplot as plt

from detection import draw_bounding_box, detect, get_values

img_path = "./20150326_185236.jpg"
image = cv2.imread(img_path)
height, width, _ = image.shape

boxes, indices, confidences, class_ids = detect(image)
centers = []
xx = []
yy = []
z = []
areas = []
# go through the detections remaining
# after nms and draw bounding box
print(indices)
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    center = [round((w/2) + x), round((h/2) + y)]
    centers.append(center)
    x_grid, y_grid, z_grid = get_values(center, width, height)
    xx.append(x_grid)
    yy.append(y_grid)
    z.append(z_grid)
    areas.append(w*h)

    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

z_sum = np.zeros_like(z[0])
for area, z_grid in zip(areas, z):
    #z_sum = z_sum + z_grid
    z_sum = z_sum + (z_grid*area) # Add weight by area

plt.imshow(image)
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
plt.plot(center_polygon[0], center_polygon[1], 'ok')
for x, y in centers:
    plt.plot(x, y, 'o')
plt.savefig("mat.png")

def adjust_values(n_0, n_f, limit_0, limit_f):
    if n_0 < limit_0:
        dif = limit_0 - n_0
        return np.floor(n_0 + dif), np.floor(n_f + dif)
    if n_f > limit_f:
        dif = n_f - limit_f
        return np.floor(n_0 - dif), np.floor(n_f - dif)
    return np.floor(n_0), np.floor(n_f)

crop_height = round(width * 2 / 3)
if crop_height > height:
    crop_height = height
    crop_width = round(height * 3 / 2)
else:
    crop_width = width

x_0 = center_polygon[0] - (crop_width/2)
x_f = center_polygon[0] + (crop_width/2)

x_0, x_f = adjust_values(x_0, x_f, 0, width)

y_0 = center_polygon[1] - (crop_height/2)
y_f = center_polygon[1] + (crop_height/2)

y_0, y_f = adjust_values(y_0, y_f, 0, height)
crop_img = image[int(y_0):int(y_f), int(x_0):int(x_f)]
#crop_img = image[int((height/2)-(crop_height/2)):int((height/2)+(crop_height/2)), 0:width]
cv2.imwrite("cropped_image.jpg", crop_img)

print("Yey")