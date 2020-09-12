import os
from PIL import Image
path = '/home/ty/data/Pre-train/DAVSOD'
file = open('/home/ty/data/Pre-train/pretrain_all_seq_DAVSOD_flow.txt', 'w')

folders = os.listdir(path)
# folders2 = os.listdir(path + '/flow')
folders.sort()
# folders2.sort()

# for folder in folders2:
#     imgs = os.listdir(os.path.join(path, 'flow', folder))
#     imgs.sort()
#     for img in imgs:
#         if img.endswith('.png'):
#             name, suffix = os.path.splitext(img)
#             old = Image.open(os.path.join(path, 'flow', folder, img))
#             new = old.save(os.path.join(path, 'flow', folder, name + '.jpg'))
#             os.remove(os.path.join(path, 'flow', folder, img))

for folder in folders:
    imgs = os.listdir(os.path.join(path, folder, 'Imgs'))
    imgs.sort()
    imgs = [os.path.splitext(img)[0] for img in imgs]
    imgs2 = os.listdir(os.path.join(path, folder, 'flow'))
    imgs2.sort()
    imgs2 = [os.path.splitext(img)[0] for img in imgs2]
    for img in imgs:
        if img in imgs2:
            img_path = os.path.join('DAVSOD', folder, 'Imgs', img + '.png')
            gt_path = os.path.join('DAVSOD', folder, 'GT_object_level', img + '.png')
            flow_path = os.path.join('DAVSOD', folder, 'flow', img + '.jpg')
            file.writelines(img_path + ' ' + flow_path + ' ' + gt_path + '\n')

file.close()