#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nodule and non-nodule classses

@author: Jarvis ZHANG
@date: 2017/10/6
@framework: Tensorflow
@editor: VS Code
"""

import numpy as np
from prettytable import PrettyTable     # download the source code and install with python setup.py

feature_names = ["subtlety",
                 "internalStructure", 
                 "calcification", 
                 "sphericity", 
                 "margin", 
                 "lobulation", 
                 "spiculation", 
                 "texture", 
                 "malignancy"]


class ROI(object):
    def __init__(self, label=None, **kwargs):
        if label is not None:
            origin = kwargs.get("origin", None)
            spacing = kwargs.get("spacing", None)
            self._create_from_lidc(label, origin, spacing)
        else:
            self._create_from_empty()


    def _create_from_lidc(self, label, origin, spacing):
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        if spacing is None:
            spacing = [1.0, 1.0, 1.0]

        self.imageZposition = int((float(label.imageZposition.text) - origin[-1]) / spacing[-1])
        self.imageSOP_UID = label.imageSOP_UID.text
        self.inclusion = bool(label.inclusion.text)
        self.edgeMap = []
        
        edgeMaps = label.find_all("edgeMap")
        for edgeMap in edgeMaps:
            x, y = int(edgeMap.xCoord.text), int(edgeMap.yCoord.text)
            self.edgeMap.append((x, y))

        self.length = len(self.edgeMap)

        if self.length > 1:
            self.isSeg = True
        else:
            self.isSeg = False


    def _create_from_empty(self):
        self.imageZposition = 0
        self.imageSOP_UID = ""
        self.inclusion = None
        self.edgeMap = []
        self.length = 0
        self.isSeg = False


    def __str__(self):
        fmt_str = ""
        fmt_str += "imageZposition: %d\n" % self.imageZposition
        fmt_str += "imageSOP_UID: %s\n" % self.imageSOP_UID
        fmt_str += "inclusion: %r\n" % self.inclusion
        fmt_str += "edgeMap: \n"
        for i in range(self.length):
            fmt_str += "(%d, %d)\n" % (self.edgeMap[i][0], self.edgeMap[i][1])
        return fmt_str


    def __getitem__(self, item):
        if item >= self.length:
            raise IndexError("Out of index!")
        return self.edgeMap[item][0], self.edgeMap[item][1], self.imageZposition


    def __eq__(self, other):
        if self.imageZposition != other.imageZposition:
            return False
        if self.length != other.length:
            return False
        for point in self.edgeMap:
            if not point in other.edgeMap:
                return False
        return True


    def __len__(self):
        return self.length


class Nodule(object):
    def __init__(self, obj=None, **kwargs):
        if obj is not None:
            origin = kwargs.get("origin", None)
            spacing = kwargs.get("spacing", None)
            self._create_from_lidc(obj, origin, spacing)
        else:
            self._create_from_empty()


    def _create_from_lidc(self, obj, origin, spacing):
        self.id = obj.noduleID.text
        self.characteristics = self.check_feature(obj)
        self.roi = []           # list of annotated layers
        self.single_anno = []   # just one annotated point in some layer
        
        rois = obj.find_all("roi")
        for roi_label in rois:
            one_roi = ROI(roi_label, origin=origin, spacing=spacing)
            if one_roi.isSeg:
                self.roi.append(one_roi)
            else:
                self.single_anno.append(one_roi)

        self.length = len(self.roi)
        self._get_center()


    def _create_from_empty(self):
        self.id = None
        self.characteristics = False
        self.feature_collection = None
        self.roi = []
        self.single_anno = []
        self.length = 0
        self._get_center()


    def __str__(self):
        fmt_str = ""
        fmt_str += "Nodule ID: %s\n" % self.id
        fmt_str += "Characteristics: %r\n" % self.characteristics
        if self.characteristics:
            pt_shape = PrettyTable()
            pt_shape._set_field_names(feature_names)
            pt_shape.add_row(self.feature_collection.values())
            fmt_str += pt_shape.get_string() + '\n'

        fmt_str += "Center coordinate: (%d, %d, %d)\n" % self.center
        fmt_str += "Diameter: %.2f\n" % (self.diameter)
        fmt_str += "ROI: \n"
        if self.length > 0:
            pt = PrettyTable()
            header = list(range(self.length + 1))
            header[0] = "layer:z"
            pt._set_field_names(header)
            pt.add_row(["z_value"] + [one_roi.imageZposition for one_roi in self.roi])
            pt.add_row(["inclusion"] + [one_roi.inclusion for one_roi in self.roi])
            x_min = ["x_min"]
            x_max = ["x_max"]
            y_min = ["y_min"]
            y_max = ["y_max"]
            for one_roi in self.roi:
                xy = list(zip(*one_roi.edgeMap))
                x_min.append(min(xy[0]))
                x_max.append(max(xy[0]))
                y_min.append(min(xy[1]))
                y_max.append(max(xy[1]))
            pt.add_row(x_min)
            pt.add_row(x_max)
            pt.add_row(y_min)
            pt.add_row(y_max)
            fmt_str += pt.get_string() + '\n'

        return fmt_str


    def __getitem__(self, item):
        if item >= self.length:
            raise IndexError("Out of index!")
        return self.roi[item]


    def __len__(self):
        return self.length


    def _get_center(self):
        x_min = y_min = z_min = 999999
        x_max = y_max = z_max = -999999

        for one_roi in self.roi:
            z_min = min(z_min, one_roi.imageZposition)
            z_max = max(z_max, one_roi.imageZposition)
            for point in one_roi.edgeMap:
                x_min = min(x_min, point[0])
                x_max = max(x_max, point[0])
                y_min = min(y_min, point[1])
                y_max = max(y_max, point[1])

        x_diameter = x_max - x_min
        y_diameter = y_max - y_min
        self.diameter = max(x_diameter, y_diameter)
        self.center = ((x_max + x_min) // 2, (y_max + y_min) // 2, (z_max + z_min) // 2)


    def check_feature(self, obj):
        if obj.characteristics is None:
            self.feature_collection = None
            return False
        
        features = obj.characteristics
        self.feature_collection = {
            feature_names[0] : int(features.subtlety.text) if features.subtlety is not None else -1,
            feature_names[1] : int(features.internalStructure.text) if features.internalStructure is not None else -1,
            feature_names[2] : int(features.calcification.text) if features.calcification is not None else -1,
            feature_names[3] : int(features.sphericity.text) if features.sphericity is not None else -1,
            feature_names[4] : int(features.margin.text) if features.margin is not None else -1,
            feature_names[5] : int(features.lobulation.text) if features.lobulation is not None else -1,
            feature_names[6] : int(features.spiculation.text) if features.spiculation is not None else -1,
            feature_names[7] : int(features.texture.text) if features.texture is not None else -1,
            feature_names[8] : int(features.malignancy.text) if features.malignancy is not None else -1
        }
        return True



class nonNodule(object):
    def __init__(self, obj, origin, spacing):
        self.id = obj.nonNoduleID.text
        self.imageZposition = (float(obj.imageZposition.text) - origin[-1]) / spacing[-1]
        self.imageSOP_UID = obj.imageSOP_UID
        x, y = int(obj.locus.xCoord.text), int(obj.locus.yCoord.text)
        self.locus = (x, y)


if __name__ == '__main__':
    nod = Nodule()
    print(nod.feature_collection)