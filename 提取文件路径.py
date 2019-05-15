import os

def readFilename(path, file):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            readFilename(filepath, file)
        else:
            file.append(filepath)
    return file


if __name__ == '__main__':
    Path = "/Users/weiyangbin/Downloads/vector_graphics_floorplans/floorplan_image/"
    file = []
    file = readFilename(Path, file)
    name = []
    txt_path = "/Users/weiyangbin/Downloads/vector_graphics_floorplans/" + "filename.txt"
    for name in file:
        print(name)
        file_cls = name.split("/")[-1].split(".")[-1]
        if file_cls == 'jpg':
            #print(name.split("/")[-1])
            with open(txt_path, 'a+') as fp:
                fp.write("".join(name) + "\n")
    # 删掉多余字符
    files = open("/Users/weiyangbin/Downloads/vector_graphics_floorplans/filename.txt")
    lis = []
    for i in files:
        lis.append([value.strip() for value in i.split('\t')])
    for i in range(len(lis)):
        lis[i][0] = lis[i][0].strip('/Users/weiyangbin/Downloads/vector_graphics_floorplans/floorplan_image/')
    for i in range(len(lis)):
        lis[i][0] = lis[i][0].strip('.j')
    with open(txt_path, 'a+') as f:
        for i in lis:
            f.write("".join(i) + "\n")







