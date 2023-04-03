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
from pathlib import Path

if __name__ == '__main__':
    FILE = Path(__file__).resolve() 
    ROOT = str(FILE.parents[0])  #demo.py 's path
    PANDA_image_path = "D:\Project\PANDA_image"
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
    1. show images or images with annos
    '''
    # example = PANDA_IMAGE(
    #     imagepath= PANDA_image_path + "\image_train",
    #     annopath= PANDA_image_path + "\image_annos\person_bbox_train.json",
    #     annomode='person'
    #     )
    
    # example.showImgs(["11_Shenzhen_Library/IMG_11_01.jpg"])
    # # or
    # example.showAnns(["11_Shenzhen_Library/IMG_11_01.jpg"])

    '''
    2. Split Image And Label
    We provide the scale param before split the images and labels.
    the splitted data path:
        └─PANDA_Toolkit
            └─split
                ├─image_annos
                    └─split.json
                └─image_train
    Before your operation,please make sure the "image_train" package is empty
    '''
    # # split the image
    # split = ImgSplit(imagepath = PANDA_image_path + '\image_train', 
    #         annopath= PANDA_image_path + '\image_annos\person_bbox_train.json', annomode="person", 
    #         outimagepath = ROOT +'\split\image_train',
    #         outannopath = ROOT + "\split\image_annos\split.json",
    #         subwidth=[1000,2000,3000],subheight=[1000,2000,3000])
    # split.splitdata(0.5,["11_Shenzhen_Library/IMG_11_01.jpg"])

    # # show the splited image and annos
    # example = PANDA_IMAGE(
    #     imagepath= ROOT + "\split\image_train",
    #     annopath= ROOT + "\split\image_annos\split.json",
    #     annomode='person'
    #     )
    # # make sure the name is right 
    # example.showAnns(["11_Shenzhen_Library_IMG_11_01___0.5__00114__13064__8300.jpg"])
    '''
    3. Merge patches
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
    indeed the name of the function is ugly,but it works.
    the merged data path:
        └─ PANDA_Toolkit
            └─ merge
                ├─ fake_result.json
                ├─ merge_COCO.json
                └─ merge_PANDA.json
    '''
    # src_anno_file = PANDA_image_path + "\image_annos\person_bbox_train.json"
    # fake_result_annos = ROOT + "\merge\\fake_result.json"
    # merge_COCO_annos = ROOT+"\merge\merge_COCO.json"
    # merge_PANDA_annos = ROOT+"\merge\merge_PANDA.json"
    # # generate fake detection results
    # util.GT2DetRes(gtpath=ROOT + "\split\image_annos\split.json", outdetpath=fake_result_annos)
    # # merge detection results
    # merge = DetResMerge(
    #     imgpath=PANDA_image_path,
    #     respath=fake_result_annos,
    #     splitannopath=ROOT + "\split\image_annos\split.json",
    #     srcannopath=src_anno_file,
    #     outfile=merge_COCO_annos
    #     )
    # # NMS
    # merge.mergeResults(is_nms=True)
    # # COCO format to PANDA format
    # util.DetRes2GT(
    #     detrespath=merge_COCO_annos, 
    #     outgtpath=merge_PANDA_annos, 
    #     gtannopath=src_anno_file
    # )

    # # show merged results
    # example = PANDA_IMAGE(imagepath=PANDA_image_path+"\image_train",annopath=merge_PANDA_annos, annomode='vehicle')
    # example.showAnns(["11_Shenzhen_Library/IMG_11_01.jpg"])