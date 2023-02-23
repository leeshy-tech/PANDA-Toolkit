# --------------------------------------------------------
# Tool kit function demonstration
# Written by Wang Xueyang (wangxuey19@mails.tsinghua.edu.cn), Version 20200523
# Inspired from DOTA dataset devkit (https://github.com/CAPTAIN-WHU/DOTA_devkit)
# --------------------------------------------------------

from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge

if __name__ == '__main__':
    '''
    Note:
    The most common runtime error is file path error. 
        For example, if you need to operate on the test set instead of the training set, 
        you need to modify the folder path in __init__ part of corresponding Class (e.g. PANDA_IMAGE) 
        from "image_train" to "image_test". 
        If you encounter other file path errors, 
        please also check the path settings in __init__ first.
    '''
    image_root = 'D:\Project\PANDA_image'
    person_anno_file = 'person_bbox_train.json'
    annomode = 'person'
    example = PANDA_IMAGE(image_root, person_anno_file, annomode='person')

    '''1. show images'''
    example.showImgs(["01_University_Canteen/IMG_01_01.jpg"])

    '''2. show annotations'''
    example.showAnns(["01_University_Canteen/IMG_01_01.jpg"])

    '''
    3. Split Image And Label
    We provide the scale param before split the images and labels.
    '''
    outpath = 'split'
    outannofile = 'split.json'
    split = ImgSplit(image_root, person_anno_file, annomode, outpath, outannofile)
    split.splitdata(0.5,["01_University_Canteen/IMG_01_01.jpg"])

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
    # groundtruth -> detection results(fake)
    util.GT2DetRes('split/image_annos/split.json', 'split/results/res.json')
    # merge detection results
    merge = DetResMerge('split', 'res.json', 'split.json', 'person_bbox_train.json', 'results', 'mergetest.json')
    # NMS
    merge.mergeResults(is_nms=True)
    # detection results -> groundtruth
    util.DetRes2GT('results/mergetest.json', 'D:\Project\PANDA_image\image_annos\mergegt.json', 
        'D:\Project\PANDA-Toolkit\split\image_annos\person_bbox_train.json')

    '''show merged results'''
    example = PANDA_IMAGE(image_root, 'mergegt.json', annomode='vehicle')
    example.showAnns(["01_University_Canteen/IMG_01_01.jpg"])