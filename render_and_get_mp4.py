import argparse
import os
import numpy as np
import cv2
#import skvideo
#skvideo.setFFmpegPath('/usr/bin/')
#import skvideo.io


# #uncompressed mp4 writer
# outputfile = "rotate.mp4"
# writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
# '-vcodec': 'libx264',  #use the h.264 codec
# '-crf': '0',           #set the constant rate factor to 0, which is lossless
# '-preset': 'ultrafast', #the slower the better compression 'veryslow'
# '-r': '24',
# },verbosity=0)


def render_colored_point_cloud(vertices,colors,size,alpha,beta,dotsize):
    #transform
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)

    raw_x = -vertices[:,0]
    raw_y = -vertices[:,1]
    raw_z = vertices[:,2]

    new_x = sin_alpha*raw_x + cos_alpha*raw_z
    new_y = raw_y
    new_z = cos_alpha*raw_x - sin_alpha*raw_z

    new_x2 = new_x
    new_y2 = cos_beta*new_y - sin_beta*new_z
    new_z2 = sin_beta*new_y + cos_beta*new_z

    order = np.argsort(-new_z2)
    px = np.clip((new_x2+1.0)*(size/2.0),0,size-dotsize).astype(np.int32)
    py = np.clip((new_y2+1.0)*(size/2.0),0,size-dotsize).astype(np.int32)
    
    img = np.full([size,size,3],255,np.uint8)
    for i in range(len(order)):
        idx = order[i]
        img[py[idx]:py[idx]+dotsize,px[idx]:px[idx]+dotsize] = colors[idx]

    return img


def read_ply_point_normal_color(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    colors = np.zeros([vertex_num,3], np.uint8)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        colors[i,0] = int(line[3])
        colors[i,1] = int(line[4])
        colors[i,2] = int(line[5])
    return vertices,colors














parser = argparse.ArgumentParser()
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the samples")
FLAGS = parser.parse_args()









sample_dir = FLAGS.sample_dir
obj_names = os.listdir(sample_dir)
#filter
part_num_at_least = 0
obj_names_ = []
obj_names_num_ = []
for name in obj_names:
    if "gt" in name or '_' not in name: continue
    num1 = int(name.split('_')[0])
    num2 = int(name.split('_')[1][:-4])
    if num2>=part_num_at_least:
        obj_names_.append(name)
        obj_names_num_.append(num2*100000+num1)
#sort based on number of contained shapes
obj_names_num_ = -np.array(obj_names_num_,np.int32)
sort_idx = np.argsort(obj_names_num_)
obj_names = []
for i in sort_idx:
    obj_names.append(obj_names_[i])
print("len:", len(obj_names))

set_points = []
set_colors = []
for i in range(len(obj_names)):
    points,colors = read_ply_point_normal_color(sample_dir+"/"+obj_names[i])
    set_points.append(points)
    set_colors.append(colors)

img_h = 1080
img_w = 1920
crop_h = 30
crop_w = 40

dotsize = 2
obj_h_num = 4
obj_w_num = 8
subimg_size = 360
subimg_text_h = 320
subimg_h = (img_h-100)//obj_h_num
subimg_w = (img_w-100)//obj_w_num
if obj_h_num*obj_w_num<len(obj_names):
    obj_names = obj_names[:obj_h_num*obj_w_num]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(sample_dir+"/"+"rotate.mp4",fourcc, 24.0, (img_w,img_h))


for r in range(360):
    if r%30==29: print(r+1,'/',360)
    alpha = r/180.0*np.pi
    beta = 0.15*np.pi

    buffer = np.full([img_h+subimg_size,img_w+subimg_size,3],255,np.uint8)
    
    for y in range(obj_h_num):
        for x in range(obj_w_num):
            idx = y*obj_w_num+x
            if idx>=len(obj_names):
                break
            img = render_colored_point_cloud(set_points[idx],set_colors[idx],subimg_size,alpha,beta,dotsize)
            img = cv2.putText(img, obj_names[idx].split('_')[1].split('.')[0], (subimg_size//2,subimg_text_h), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
            buffer[y*subimg_h:y*subimg_h+subimg_size,x*subimg_w:x*subimg_w+subimg_size] = np.minimum(buffer[y*subimg_h:y*subimg_h+subimg_size,x*subimg_w:x*subimg_w+subimg_size], img)
    
    buffer = buffer[crop_h:crop_h+img_h,crop_w:crop_w+img_w]

    # writer.writeFrame(buffer)
    writer.write(buffer)


# writer.close()
writer.release()
