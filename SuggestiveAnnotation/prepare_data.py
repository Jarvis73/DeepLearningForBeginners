#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process LIDC labels

@author: Jarvis ZHANG
@date: 2017/10/5
@framework: Tensorflow
@editor: VS Code
"""

#import tensorflow as tf
#import helpers
import os
import numpy as np
from numpy.linalg import norm
import Nodule
import ntpath
import allPath
import SimpleITK
from glob import glob
from bs4 import BeautifulSoup


def find_mhd_file(patient_id):
    for src_path in glob(ntpath.join(allPath.LUNA16_RAW_SRC_DIR, "*.mhd")):
        if patient_id in src_path:
            return src_path
    return None


def merge_two_nodules(n1, n2):
    """ Merge two coincident nodules annotated by two different radiologists 
    
    ### Params:
        * n1: Nodule object 1
        * n2: Nodule object 2
    
    ### Returns:
        * merged: Merged nodule
    """

    merged = Nodule.Nodule()
    merged.id = n1.id + "&" + n2.id

    # merge characteristics
    merged.characteristics = True if n1.characteristics or n2.characteristics else False
    if merged.characteristics:
        if n1.characteristics and not n2.characteristics:
            merged.feature_collection = n1.feature_collection
        elif not n1.characteristics and n2.characteristics:
            merged.feature_collection = n2.feature_collection
        else:
            merged.feature_collection = {}
            for key, value in n1.feature_collection.items():
                merged.feature_collection[key] = (n2.feature_collection[key] + value) / 2

    # merge rois
    n1_roi_z = [roi1.imageZposition for roi1 in n1.roi]
    n2_roi_z = [roi2.imageZposition for roi2 in n2.roi]
    for roi2 in n2.roi:
        if not roi2.imageZposition in n1_roi_z:
            merged.roi.append(roi2)
        else:
            for roi1 in n1.roi:
                if roi1.imageZposition == roi2.imageZposition:
                    merged_roi = list(set(roi1.edgeMap) | set(roi2.edgeMap))
                    merged.roi.append(merged_roi)
                    break
    for roi1 in n1.roi:
        if not roi1.imageZposition in n2_roi_z:
            merged.roi.append(roi1)

    # merge single_anno
    n1_sig_z = [sig1.imageZposition for sig1 in n1.single_anno]
    n2_sig_z = [sig2.imageZposition for sig2 in n2.single_anno]
    for sig2 in n2.single_anno:
        if not sig2.imageZposition in n1_sig_z:
            merged.single_anno.append(sig2)
        else:
            for sig1 in n1.single_anno:
                if sig1.imageZposition == sig2.imageZposition:
                    merged_sig = list(set(sig1.edgeMap) | set(sig2.edgeMap))
                    merged.single_anno.append(merged_sig)
                    break
    for sig in n1.single_anno:
        if not sig1.imageZposition in n2_sig_z:
            merged.single_anno.append(sig1)

    merged.length = len(merged.roi)
    merged.center = merged._get_center()

    return merged


def load_lidc_xml(xml_path, threshold=0):
    with open(xml_path, "r") as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    src_path = find_mhd_file(patient_id)
    if src_path is None:
        return None
    
    itk_image = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_image)
    num_z, height, width = img_array.shape        #height X width constitute the transverse plane
    origin = np.array(itk_image.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_image.GetSpacing())    # spacing of voxels in world coor. (mm)

    # Find all the radiologists
    all_nodules = []
    readingSessions = xml.LidcReadMessage.find_all("readingSession")
    for radio, readingSession in enumerate(readingSessions):
        # Find all nodules
        nodules = readingSession.find_all("unblindedReadNodule")
        nodules = [Nodule.Nodule(nod, origin=origin, spacing=spacing) for nod in nodules]
        all_nodules += nodules
        # # Dump
        # for i, nodule in enumerate(nodules):
        #     if nodule.roi != []:
        #         with open(ntpath.join(allPath.SA_SEG_DIR, "T_" + patient_id + "_%d%d.txt" % (radio, i)), "w") as f:
        #             for roi in nodule:
        #                 for point in roi:
        #                     f.write(str(point) + "\n")

        if False:
            # Fine all non-nodules
            nonNodules = readingSession.find_all("nonNodule")
            nonNodules = [Nodule.nonNodule(nonNod, origin=origin, spacing=spacing) for nonNod in nonNodules]

    # screening
    if threshold > 0:
        filtered_nodules = []
        for n1 in all_nodules:
            if n1.length == 0:
                continue
            for n2 in all_nodules:
                if n2.length == 0 or n1.id == n2.id:
                    continue
                dist = norm(np.array(n1.center) - np.array(n2.center))
                if dist < n1.diameter or dist < n2.diameter:
                    pass # Merge












def process_lidc_segmentation(log=True):
    if not ntpath.exists(allPath.LIDC_XML_DIR):
        print("Fatal error:", allPath.LIDC_XML_DIR, "not found!")
        return
    
    file_no = 0
    all_dirs = [d for d in glob(ntpath.join(allPath.LIDC_XML_DIR, "*")) if ntpath.isdir(d)]
    for anno_dir in all_dirs:
        xml_paths = glob(ntpath.join(anno_dir, "*.xml"))
        for xml_path in xml_paths:
            if log:
                print(file_no, ":", xml_path)
            load_lidc_xml(xml_path)
            file_no += 1
            if file_no > 1:
                return

    

if __name__ == '__main__':
    if True:
        process_lidc_segmentation()





