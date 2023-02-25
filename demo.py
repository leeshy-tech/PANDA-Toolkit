# --------------------------------------------------------
# Tool kit function demonstration
# Modified by LiSai (saili@bupt.edu.cn), version 20230224
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge

def show_image(image_root,annofile,annomode,image_name,annos=False):
    example = PANDA_IMAGE(image_root, annofile, annomode='person')
    if annos:
        example.showAnns([image_name])
    else:
        example.showImgs([image_name])

def split_images(image_root,image_list,anno_file,subwidth=2048,subheight=1024):
    split = ImgSplit(image_root, anno_file, annomode="person", outpath='split', 
                    outannofile='split.json',subwidth=subwidth,subheight=subheight)
    split.splitdata(0.5,image_list)


if __name__ == '__main__':
    '''
    Note:
        you should download the PANDA dataset(https://www.gigavision.cn/track/track/?nav=Detection),
        include: image_annos,image_test,image_train
        It is recommended to organize it as:
            └─PANDA_image
                ├─image_annos
                ├─image_test
                └─image_train
    '''
    '''
    show images or images with annos
    '''
    # show_image(image_root='D:\Project\PANDA_image', 
    #     annofile='person_bbox_train.json', 
    #     annomode='person', 
    #     image_name='01_University_Canteen/IMG_01_01.jpg',
    #     annos=True)

    # show_image(image_root='D:\Project\PANDA-Toolkit\split', 
    #     annofile='split.json', 
    #     annomode='person', 
    #     image_name='01_University_Canteen_IMG_01_01___0.5__2772__2772.jpg',
    #     annos=True)

    '''
    3. Split Image And Label
    We provide the scale param before split the images and labels.
    the splitted data path:
        └─PANDA_Toolkit
            └─split
                ├─image_annos
                    └─split.json
                └─image_train
    Before your operation,please make sure the "image_train" package is empty
    '''
    # split_images(image_root='D:\Project\PANDA_image', 
    #     image_list=["01_University_Canteen/IMG_01_01.jpg"],
    #     anno_file='person_bbox_train.json',
    #     subwidth=1024,
    #     subheight=1024)
    '''
    4. Merge patches
    Now, we will merge these patches to see if they can be restored in the initial large images
    '''
    '''
    Note:
    GT2DetRes is used to generate 'fake' detection results (only visible body BBoxes are used) from ground-truth annotation. 
        That means, GT2DetRes is designed to generate some intermediate results to demostrate functions. 
        And in practical use, you doesn't need to use GT2DetRes because you 
        have real detection results file on splited images and you can merge them using DetResMerge.
    DetRes2GT is used to transfer the file format from COCO detection result file to PANDA annotation file 
        in order to visualize detection results using PANDA_IMAGE apis. 
        Noted that DetRes2GT is not yet fully designed and can only transfer objects from single category 
        (visible body). If you have other requirements, please make your own changes.
    '''
    # generate fake detection results by GT
    util.GT2DetRes('split/image_annos/split.json', 'split/results/res.json')
    # merge detection results
    merge = DetResMerge('split', 'res.json', 'split.json', 'person_bbox_train.json', 'results', 'mergetest.json')
    # NMS
    merge.mergeResults(is_nms=True)
    # detection results -> groundtruth
    util.DetRes2GT('results/mergetest.json', 'D:\Project\PANDA_image\image_annos\mergegt.json', 
        'D:\Project\PANDA-Toolkit\split\image_annos\person_bbox_train.json')

    '''show merged results'''
    example = PANDA_IMAGE('D:\Project\PANDA_image', 'mergegt.json', annomode='vehicle')
    example.showAnns(["01_University_Canteen/IMG_01_01.jpg"])