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
import pprint
import SimpleITK
from glob import glob
from bs4 import BeautifulSoup


def find_mhd_file(patient_id):
    for src_path in glob(ntpath.join(allPath.LUNA16_RAW_SRC_DIR, "*.mhd")):
        if patient_id in src_path:
            return src_path
    return None


def merge_two_roi(r1, r2):
    """ Merge two coincident rois
    
    ### Params:
        * r1: ROI object 1
        * r2: ROI object 2
    
    ### Returns:
        * merged: Merged ROI
    """

    assert(r1.imageZposition == r2.imageZposition)
    merged = Nodule.ROI()
    merged.imageZposition = r1.imageZposition
    merged.imageSOP_UID = "Merged: disabled UID"
    merged.inclusion = True if r1.inclusion or r2.inclusion else False
    merged.edgeMap += list(set(r1.edgeMap) | set(r2.edgeMap))
    merged.length = len(merged.edgeMap)
    merged.isSeg = True if r1.isSeg or r2.isSeg else False
    return merged


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
                    merged_roi = merge_two_roi(roi1, roi2)
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
                    merged_sig = merge_two_roi(sig1, sig2)
                    merged.single_anno.append(merged_sig)
                    break
    for sig1 in n1.single_anno:
        if not sig1.imageZposition in n2_sig_z:
            merged.single_anno.append(sig1)

    merged.length = len(merged.roi)
    merged._get_center()

    return merged


def load_lidc_xml(xml_path, threshold=-1, screening=True, process_only_patient=None):
    with open(xml_path, "r") as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")

    # IDRI nodules are ignored
    if xml.LidcReadMessage is None: 
        print("IDRI")
        return False

    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    if process_only_patient is not None and patient_id != process_only_patient:
        print("pass")
        return False

    # Not included in Luna16 datasets
    src_path = find_mhd_file(patient_id)
    if src_path is None:
        print("Not Luna16")
        return False
    
    # Get origin and spacing which are used to normalize z coordinate
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
        all_nodules += nodules  # just add all the annotated nodules to all_nodules

        if False:   # ignore
            # Fine all non-nodules
            nonNodules = readingSession.find_all("nonNodule")
            nonNodules = [Nodule.nonNodule(nonNod, origin=origin, spacing=spacing) for nonNod in nonNodules]

    # screening: 1. ignore single annotated point(length=0)
    #            2. merge same nodules
    if screening:
        filtered_nodules = []
        all_set = set(range(len(all_nodules)))
        for i, n1 in enumerate(all_nodules):
            if i not in all_set:
                continue
            all_set.remove(i)   # compare nodule i and nodule j (j > i)

            if n1.length == 0:  # single annotated point
                continue

            overlaps = 0    # counter: how many nodules overlapping with i_th nodule
            for j, n2 in enumerate(all_nodules):
                if j not in all_set:
                    continue

                if n2.length == 0:  # single annotated point
                    continue
                
                dist = norm(np.array(n1.center) - np.array(n2.center))
                if dist < n1.diameter or dist < n2.diameter:    # same nodules
                    overlaps += 1
                    all_set.remove(j)   # merged nodules should not be check again
                    n1 = merge_two_nodules(n1, n2)
            
            # If there is not threshold, then at least 0 > -1; else overlaps > threshold
            if overlaps >= threshold:
                filtered_nodules.append(n1) # end for j, n2 in enumerate(all_nodules)

        all_nodules = filtered_nodules  # end for i, n1 in enumerate(all_nodules)

    print("\n" + patient_id)
    for nodule in all_nodules:
        print(nodule.center, nodule.diameter)

    return True


def process_lidc_segmentation(log=True, process_only_patient=None):
    if not ntpath.exists(allPath.LIDC_XML_DIR):
        print("Fatal error:", allPath.LIDC_XML_DIR, "not found!")
        return
    
    file_no = 0
    all_dirs = [d for d in glob(ntpath.join(allPath.LIDC_XML_DIR, "*")) if ntpath.isdir(d)]
    for anno_dir in all_dirs:
        xml_paths = glob(ntpath.join(anno_dir, "*.xml"))
        for xml_path in xml_paths:
            if log:
                print(file_no, ":", xml_path, ": ", end='')
            if load_lidc_xml(xml_path, threshold=1, process_only_patient=process_only_patient) \
                    and process_only_patient is not None:
                return
            file_no += 1

    

if __name__ == '__main__':
    if True:
        process_only_patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.214252223927572015414741039150"
        process_lidc_segmentation(process_only_patient=None)





