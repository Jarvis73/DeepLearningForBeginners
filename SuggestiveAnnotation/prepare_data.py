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
import Nodule
import os
from glob import glob
import ntpath
from bs4 import BeautifulSoup

LUNA16_RAW_SRC_DIR = "C:\\DataSet\\Luna16\\data_set"
LIDC_XML_DIR = "C:\\DataSet\\Luna16\\LIDC-XML-only\\tcia-lidc-xml"


def find_mhd_file(patient_id):
    for src_path in glob(ntpath.join(LUNA16_RAW_SRC_DIR, "*.mhd")):
        if patient_id in src_path:
            return src_path
    return None


def load_lidc_xml(xml_path, threshold=None):
    with open(xml_path, "r") as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    src_path = find_mhd_file(patient_id)
    if src_path is None:
        return None

    # Find all the radiologists
    readingSessions = xml.LidcReadMessage.find_all("readingSession")
    for readingSession in readingSessions:
        # Find all nodules
        nodules = readingSession.find_all("unblindedReadNodule")
        nodules = [Nodule.Nodule(nod) for nod in nodules]
        for nodule in nodules:
            print(nodule)
            return

        # Fine all non-nodules
        nonNodules = readingSession.find_all("nonNodule")
        nonNodules = [Nodule.nonNodule(nonNod) for nonNod in nonNodules]





def process_lidc_segmentation(log=True):
    if not ntpath.exists(LIDC_XML_DIR):
        print("Fatal error:", LIDC_XML_DIR, "not found!")
        return
    
    file_no = 0
    all_dirs = [d for d in glob(ntpath.join(LIDC_XML_DIR, "*")) if ntpath.isdir(d)]
    for anno_dir in all_dirs:
        xml_paths = glob(ntpath.join(anno_dir, "*.xml"))
        for xml_path in xml_paths:
            if log:
                print(file_no, ":", xml_path)
            load_lidc_xml(xml_path)
            pass

    

if __name__ == '__main__':
    if True:
        process_lidc_segmentation()





