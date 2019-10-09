import os

for a,b,c in os.walk(top="/home/liupeng/workspace/Temporal_2D/data/images/test"):
    images = c
    for i, image in enumerate(c):
        if 'jpg' not in image:
            continue
        name = int(image.split('.')[0][-4:])
        name = str(int(name/5))
        name = ''.join(['0' for _ in range(4-len(name))]+list(name))
        os.rename(a+'/'+c[i], a+'/'+name+'.jpg')

