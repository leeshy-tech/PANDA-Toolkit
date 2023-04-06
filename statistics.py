import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.ticker import PercentFormatter

if __name__ == '__main__':
    save_path = "stat_result"
    person_annopath = "D:\Project\PANDA_image\image_annos\\person_bbox_train.json"
    vehicle_annopath = "D:\Project\PANDA_image\image_annos\\vehicle_bbox_train.json"

    scene_list = [
        "01_University_Canteen",
        "02_Xili_Crossroad",
        "03_Train_Station Square",
        "04_Grant_Hall",
        "05_University_Gate",
        "06_University_Campus",
        "07_East_Gate",
        "08_Dongmen_Street",
        "09_Electronic_Market",
        "10_Ceremony",
        "11_Shenzhen_Library",
        "12_Basketball_Court",
        "13_University_Playground",
        "all"
    ]
    vehicle_list = [
        "small car",
        "midsize car",
        "large car",
        "bicycle",
        "motorcycle",
        "tricycle",
        "electric car",
        "baby carriage"
    ]

    for scene in scene_list:
        # person
        person_hlist = []
        person_ylist = []
        person_xlist = []
        with open(person_annopath, 'r') as load_f:
            annodict = json.load(load_f)
            for (imagename, imagedict) in annodict.items():
                image_scene = imagename.split("/")[0]
                if image_scene != scene and scene != "all":
                    continue
                height = imagedict["image size"]["height"]
                objlist = imagedict["objects list"]
                for obj in objlist:
                    if obj["category"] == "person":
                        person_height = (obj["rects"]["full body"]["br"]["y"] - obj["rects"]["full body"]["tl"]["y"]) * height
                        person_y = (obj["rects"]["full body"]["br"]["y"] + obj["rects"]["full body"]["tl"]["y"]) / 2
                        person_x = (obj["rects"]["full body"]["br"]["x"] + obj["rects"]["full body"]["tl"]["x"]) / 2
                        person_hlist.append(person_height)
                        person_ylist.append(person_y)
                        person_xlist.append(person_x)
                    elif obj["category"] == "people":
                        person_height = (obj["rect"]["br"]["y"] - obj["rect"]["tl"]["y"]) * height
                        person_y = (obj["rect"]["br"]["y"] + obj["rect"]["tl"]["y"]) / 2
                        person_x = (obj["rect"]["br"]["x"] + obj["rect"]["tl"]["x"]) / 2
                        person_hlist.append(person_height)
                        person_ylist.append(person_y)
                        person_xlist.append(person_x)
        # vehicle
        vehicle_size_list = []
        vehicle_ylist = []
        vehicle_xlist = []
        with open(vehicle_annopath, 'r') as load_f:
            annodict = json.load(load_f)
            for (imagename, imagedict) in annodict.items():
                image_scene = imagename.split("/")[0]
                if image_scene != scene and scene != "all":
                    continue
                height = imagedict["image size"]["height"]
                width = imagedict["image size"]["width"]
                objlist = imagedict["objects list"]
                for obj in objlist:
                    if obj["category"] in vehicle_list:
                        h = (obj["rect"]["br"]["y"] - obj["rect"]["tl"]["y"]) * height
                        w = (obj["rect"]["br"]["x"] - obj["rect"]["tl"]["x"]) * width
                        vehicle_size = max(h,w)
                        vehicle_y = (obj["rect"]["br"]["y"] + obj["rect"]["tl"]["y"]) / 2
                        vehicle_x = (obj["rect"]["br"]["x"] + obj["rect"]["tl"]["x"]) / 2
                        vehicle_size_list.append(vehicle_size)
                        vehicle_ylist.append(vehicle_y)
                        vehicle_xlist.append(vehicle_x)

        person_array = np.array(person_hlist)
        vehicle_array = np.array(vehicle_size_list)
        person_ylist = np.array(person_ylist)
        vehicle_ylist = np.array(vehicle_ylist)
        person_xlist = np.array(person_xlist)
        vehicle_xlist = np.array(vehicle_xlist)

        # pdf
        plt.hist(person_array,bins=20,facecolor='limegreen',label="person", alpha=0.5,weights=np.ones(len(person_array)) / len(person_array))
        plt.hist(vehicle_array,bins=20,facecolor='orange',label="vehicle", alpha=0.5,weights=np.ones(len(vehicle_array)) / len(vehicle_array))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("pixel")

        plt.title(scene)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{scene}_pdf.png")
        plt.clf()
        # cdf
        plt.hist(person_array, bins=40, facecolor='limegreen', label="person", density=True, histtype='step', cumulative=True)
        plt.hist(vehicle_array, bins=40, facecolor='orange', label="vehicle", density=True, histtype='step', cumulative=True)
        plt.xlabel("pixel")

        plt.title(scene)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{scene}_cdf.png")
        plt.clf()

        # Size distribution on the y axis
        plt.scatter(person_ylist, person_array,s=5,label="person")
        plt.scatter(vehicle_ylist, vehicle_array,s=5,label="vehicle")
        plt.xlabel("center y (normalized)")
        plt.ylabel("object size")

        plt.title(scene)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{scene}_size_of_y.png")
        plt.clf()

        # Size distribution on the x axis
        plt.scatter(person_xlist, person_array,s=5,label="person")
        plt.scatter(vehicle_xlist, vehicle_array,s=5,label="vehicle")
        plt.xlabel("center x (normalized)")
        plt.ylabel("object size")

        plt.title(scene)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/{scene}_size_of_x.png")
        plt.clf()