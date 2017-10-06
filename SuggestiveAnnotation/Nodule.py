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
from prettytable import PrettyTable

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
    def __init__(self, label):
        self.imageZposition = float(label.imageZposition.text)
        self.imageSOP_UID = label.imageSOP_UID.text
        self.inclusion = bool(label.inclusion.text)
        self.edgeMap = []
        
        edgeMaps = label.find_all("edgeMap")
        for edgeMap in edgeMaps:
            x, y = int(edgeMap.xCoord.text), int(edgeMap.yCoord.text)
            self.edgeMap.append((x, y))

        if len(self.edgeMap) > 1:
            self.isSeg = True
        else:
            self.isSeg = False



class Nodule(object):
    def __init__(self, obj):
        self.id = obj.noduleID.text
        self.characteristics = self.check_feature(obj)
        self.roi = []           # list of annotated layers
        self.single_anno = []   # just one annotated point in some layer
        
        rois = obj.find_all("roi")
        for roi_label in rois:
            one_roi = ROI(roi_label)
            if one_roi.isSeg:
                self.roi.append(one_roi)
            else:
                self.single_anno.append(one_roi)

        self._get_center()


    def __str__(self):
        fmt_str = ""
        fmt_str += "Nodule ID: %s\n" % self.id
        fmt_str += "Characteristics: %r\n" % self.characteristics
        if self.characteristics:
            pt_shape = PrettyTable()
            pt_shape._set_field_names(feature_names)
            pt_shape.add_row(self.feature_collection.values())
            fmt_str += pt_shape.get_string()

        fmt_str += "\nCenter coordinate: (%d, %d, %.2f)\n" % (self.x_center, self.y_center, self.z_center)
        fmt_str += "Diameter: %.2f\n" % (self.diameter)
        fmt_str += "ROI: \n"
        pt = PrettyTable()
        header = list(range(len(self.roi) + 1))
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
        return fmt_str + pt.get_string()


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
        self.x_center = (x_max + x_min) / 2
        self.y_center = (y_max + y_min) / 2
        self.z_center = (z_max + z_min) / 2


    def check_feature(self, obj):
        if obj.characteristics is None:
            return False
        
        features = obj.characteristics
        self.feature_collection = {
            feature_names[0] : int(features.subtlety.text),
            feature_names[1] : int(features.internalStructure.text),
            feature_names[2] : int(features.calcification.text),
            feature_names[3] : int(features.sphericity.text),
            feature_names[4] : int(features.margin.text),
            feature_names[5] : int(features.lobulation.text),
            feature_names[6] : int(features.spiculation.text),
            feature_names[7] : int(features.texture.text),
            feature_names[8] : int(features.malignancy.text),
        }
        return True



class nonNodule(object):
    def __init__(self, obj):
        self.id = obj.nonNoduleID.text
        self.imageZposition = float(obj.imageZposition.text)
        self.imageSOP_UID = obj.imageSOP_UID
        x, y = int(obj.locus.xCoord.text), int(obj.locus.yCoord.text)
        self.locus = (x, y)

