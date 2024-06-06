from pycocotools.coco import COCO  
import numpy as np 
import skimage.io as io 
import matplotlib.pyplot as plt 
import pylab # matplotlib的一个模块，用于二维、三维图像绘制
pylab.rcParams['figure.figsize'] = (8.0,10.0) # 设置画布大小


annFile = './data/sim_mask/2023_10_13-11_38_10-no_wrapper/annotation.json'
imagedir = './data/sim_mask/2023_10_13-11_38_10-no_wrapper/images'
coco = COCO(annFile)

catIds = coco.getCatIds()
#print("The total number of categories: \n",len(catIds))
#print("Categories Ids: \n",catIds)

cats = coco.loadCats(coco.getCatIds()) # [{'supercategory': 'person', 'id': 1, 'name': 'person'},...]
#print("Categories Names: \n",cats)

nms = [cat['name'] for cat in cats]
print("COCO categories: \n",nms)

imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread('%s/%s'%(imagedir,img['file_name']))
#使用url加载图像
#I = io.imread(img['coco_url']) # 这应该是到网络上加载图像

#plt.axis('off') #不显示坐标尺寸


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'],catIds=catIds,iscrowd=None) #catIds用于显示指定类别的标签
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

plt.show()