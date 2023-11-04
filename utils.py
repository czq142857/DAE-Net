import numpy as np

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
    normals = np.zeros([vertex_num,3], np.float32)
    colors = np.zeros([vertex_num,3], np.int32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        normals[i,0] = float(line[3])
        normals[i,1] = float(line[4])
        normals[i,2] = float(line[5])
        colors[i,0] = int(line[6])
        colors[i,1] = int(line[7])
        colors[i,2] = int(line[8])
    return vertices,normals,colors

def write_ply_point(output_dir,vertices):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+"\n")
    fout.close()

def write_ply_point_normal(output_dir,vertices,normals):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property float nx\n" +
                "property float ny\n" +
                "property float nz\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(normals[i,0])+" "+str(normals[i,1])+" "+str(normals[i,2])+"\n")
    fout.close()

def write_ply_point_color(output_dir,vertices,colors):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    fout.close()

def write_ply_point_normal_color(output_dir,vertices,normals,colors):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property float nx\n" +
                "property float ny\n" +
                "property float nz\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(normals[i,0])+" "+str(normals[i,1])+" "+str(normals[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    fout.close()

def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "element face "+str(len(triangles))+"\n" +
                "property list uchar int vertex_index\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+"\n")
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
    fout.close()

def write_ply_triangle_color(name, vertices, colors, triangles):
    fout = open(name, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "element face "+str(len(triangles))+"\n" +
                "property list uchar int vertex_index\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
    fout.close()


#.txt format  --  X,Y,Z, normalX,normalY,normalZ, label
def parse_txt_points(shape_name,labels_unique):
    #open file & read points
    file = open(shape_name, 'r')
    lines = file.readlines()
    file.close()

    points = []
    labels = []
    for i in range(len(lines)):
        line = lines[i].split()
        points.append([float(line[2]),float(line[1]),-float(line[0])])
        labels.append(int(float(line[6])))

    point_num = len(labels)
    shape_points = np.array(points, np.float32)
    shape_labels = np.zeros([point_num], np.int32)

    for i in range(point_num):
        shape_labels[i] = labels_unique.index(labels[i])

    return shape_points, shape_labels, point_num


def get_list_of_labels(txt_name):
    #open file & read points
    file = open(txt_name, 'r')
    lines = file.readlines()
    file.close()
    labels = []
    for i in range(len(lines)):
        line = lines[i].split()
        labels.append(int(float(line[6])))
    return labels


def parse_txt_list(ref_txt_name, data_dir):
    #open file & read points
    ref_file = open(ref_txt_name, 'r')
    ref_names = [line.strip() for line in ref_file]
    ref_file.close()

    num_shapes = len(ref_names)
    point_num_max = 3000

    labels = []
    for i in range(num_shapes):
        shape_name = data_dir+"/"+ref_names[i]+".txt"
        labels += get_list_of_labels(shape_name)
    labels_unique = list(np.unique(labels))
    labels_unique = sorted(labels_unique)
    part_num = len(labels_unique)

    ref_points = np.zeros([num_shapes,point_num_max,3], np.float32)
    ref_labels = np.zeros([num_shapes,point_num_max], np.int32)
    ref_point_num = np.zeros([num_shapes], np.int32)

    for i in range(num_shapes):
        shape_name = data_dir+"/"+ref_names[i]+".txt"
        shape_points, shape_labels, point_num = parse_txt_points(shape_name,labels_unique)

        ref_points[i,:point_num] = shape_points
        ref_labels[i,:point_num] = shape_labels
        ref_point_num[i] = point_num

    return ref_points, ref_labels, ref_point_num, part_num, ref_names