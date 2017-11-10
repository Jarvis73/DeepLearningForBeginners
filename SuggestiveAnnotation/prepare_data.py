#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process LIDC labels

@author: Jarvis ZHANG
@date: 2017/10/5
@framework: Tensorflow
@editor: VS Code
"""

# import tensorflow as tf
# import helpers
import os
import cv2
import numpy as np
from numpy.linalg import norm
import Nodule
import allPath
import pandas
import SimpleITK
from glob import glob
from bs4 import BeautifulSoup
import tensorflow as tf
from skimage.morphology import binary_closing, disk, binary_erosion
from skimage.filters import roberts
from scipy.ndimage import binary_fill_holes


def find_mhd_file(patient_id):
    for src_path in glob(os.path.join(allPath.LUNA16_RAW_SRC_DIR, "*.mhd")):
        if patient_id in src_path:
            return src_path
    return None


def merge_two_roi(r1, r2):
    """ Merge two coincident rois

    # Params:
        * r1: ROI object 1
        * r2: ROI object 2

    # Returns:
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

    # Params:
        * n1: Nodule object 1
        * n2: Nodule object 2

    # Returns:
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
                merged.feature_collection[key] = (
                    n2.feature_collection[key] + value) / 2

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


def load_lidc_xml(log, xml_path, 
                threshold=-1, 
                screening=True, 
                dump_annos=True,
                dump_chars=False,
                process_only_patient=None,
                **kwargs):
    with open(xml_path, "r") as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")

    # IDRI nodules are ignored
    if xml.LidcReadMessage is None:
        if log:
            print("IDRI")
        return 1

    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    if process_only_patient is not None and patient_id != process_only_patient:
        # print("pass")
        return 1

    # Not included in Luna16 datasets
    src_path = find_mhd_file(patient_id)
    if src_path is None:
        if log:
            print("Not Luna16")
        return 1

    # Get origin and spacing which are used to normalize z coordinate
    itk_image = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_image)
    # height X width constitute the transverse plane
    num_z, height, width = img_array.shape
    # x,y,z  Origin in world coordinates (mm)
    origin = np.array(itk_image.GetOrigin())
    # spacing of voxels in world coor. (mm)
    spacing = np.array(itk_image.GetSpacing())

    # Find all the radiologists
    all_nodules = []
    readingSessions = xml.LidcReadMessage.find_all("readingSession")
    for radio, readingSession in enumerate(readingSessions):
        # Find all nodules
        nodules = readingSession.find_all("unblindedReadNodule")
        nodules = [Nodule.Nodule(nod, origin=origin, spacing=spacing)
                                 for nod in nodules]
        all_nodules += nodules  # just add all the annotated nodules to all_nodules

        if False:   # ignore

            # Fine all non-nodules
            nonNodules = readingSession.find_all("nonNodule")
            nonNodules = [Nodule.nonNodule(
                nonNod, origin=origin, spacing=spacing) for nonNod in nonNodules]


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
                    # merged nodules should not be check again
                    all_set.remove(j)
                    n1 = merge_two_nodules(n1, n2)

            # If there is not threshold, then at least 0 > -1; else overlaps > threshold
            if overlaps >= threshold:
                # end for j, n2 in enumerate(all_nodules)
                filtered_nodules.append(n1)

        # Get all segmentated nodules, members of list are Nodule objects
        # end for i, n1 in enumerate(all_nodules)
        all_nodules = filtered_nodules

    print(len(all_nodules))
    # for nodule in all_nodules:
    #     print(nodule)

    if dump_chars:      # Dump all the characteristics
        chars_list = kwargs.get('chars_list', None)
        if chars_list is None:
            print("Invalid keyword parameter: chars_list")
            return 2
        
        if not isinstance(chars_list, list):
            raise TypeError("Wrong parameter type: chars_list should be a list.")

        n = 0
        for nod in all_nodules:
            if nod.characteristics:
                chars_list.append([patient_id, n] + list(nod.feature_collection.values()))
            n += 1


    if dump_annos:      # Dump all the annotations
        n = 0
        for nod in all_nodules:
            write_path = os.path.join(
                allPath.SA_SEG_DIR, "T_%s_%d.txt" % (patient_id, n))
            with open(write_path, "w") as f:
                for i, roi in enumerate(nod):
                    for j in range(len(roi)):
                        f.write(str(nod[i][j]) + '\n')
            n += 1

    return 0


def process_lidc_segmentation(log=True, 
                            screening=True,
                            dump_annos=True,
                            dump_chars=False,
                            process_only_patient=None, 
                            begin_with_No=0):
    if not os.path.exists(allPath.LIDC_XML_DIR):
        print("Fatal error:", allPath.LIDC_XML_DIR, "not found!")
        return

    file_no = 0
    all_dirs = [d for d in glob(os.path.join(allPath.LIDC_XML_DIR, "*")) if os.path.isdir(d)]

    chars_list = []
    flag = -1

    for anno_dir in all_dirs:
        xml_paths = glob(os.path.join(anno_dir, "*.xml"))
        for xml_path in xml_paths:
            if file_no < begin_with_No:
                file_no += 1
                continue

            if log:
                print(file_no, ":", xml_path, ": ")

            flag = load_lidc_xml(
                log,
                xml_path, 
                threshold=1,
                screening=screening,
                dump_annos=dump_annos,
                dump_chars=dump_chars, 
                process_only_patient=process_only_patient,
                chars_list=chars_list
            )
            if flag == 0 and process_only_patient is not None:
                break
            elif flag == 2:
                return

            file_no += 1
        if flag == 0 and process_only_patient is not None:
            break

    if dump_chars:
        out_path = os.path.join(allPath.SA_ROOT_DIR, "characteristic.csv")
        df_chars = pandas.DataFrame(chars_list, columns=["patient_id", "nodule_id"] + Nodule.feature_names)
        df_chars.to_csv(out_path, index=False)


def rescale_patient_images(images_zyx, org_spacing_xyz, final_spacing):
    # Resample in Z axis
    resize_x = 1.0  # Fix x and change z
    resize_y = float(org_spacing_xyz[2]) / float(final_spacing)
    interpolation = cv2.INTER_LINEAR
    # Opencv assumes image shape [y, x, channels]
    # Make sure that dimentations of x is not greater than 512
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x,
                     fy=resize_y, interpolation=interpolation)

    res = res.transpose(1, 2, 0)
    # Resample in XY axis
    resize_x = float(org_spacing_xyz[0]) / float(final_spacing)
    resize_y = float(org_spacing_xyz[1]) / float(final_spacing)
    # cv2 can handle max 512 channels
    if res.shape[2] > 512:
        res1 = res[..., :512]
        res2 = res[..., 512:]
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res = np.dstack([res1, res2])
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.transpose(2, 0, 1)
    return res


def get_cube_from_image(image_zyx, center, cube_size):
    """ Extract cube from 3D image

    ### Params:
        * image_zyx: 3D array, base point is in the top left corner
        * center: list with 3 elements, center coordinates of nodule. Base point is in the top right corner
        * cube_size: integer, cube size

    ### Returns:
        * res: 3D array, the extracted cube
    """

    start_x = max(center[0] - cube_size / 2, 0)
    if start_x + cube_size > image_zyx.shape[2]:
        start_x = image_zyx.shape[2] - cube_size
    start_y = max(center[1] - cube_size / 2, 0)
    if start_y + cube_size > image_zyx.shape[1]:
        start_y = image_zyx.shape[1] - cube_size
    start_z = max(center[2] - cube_size / 2, 0)
    if start_z + cube_size > image_zyx.shape[0]:
        start_z = image_zyx.shape[0] - cube_size
    start_z = int(round(start_z))
    start_y = int(round(start_y))
    start_x = int(round(start_x))
    res = image_zyx[start_z:start_z + cube_size, start_y:start_y + cube_size, start_x:start_x + cube_size]
    return res, (start_x, start_y, start_z)


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def save_cube_img(target_path, cube_img, rows, cols):
    '''
        存为8行8列的分块矩阵, 总共64块分别为cube的64个切片
    '''
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[2]
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def z_interpolation(img, z1, z2):
    two_layer = np.dstack((img[z1], img[z2]))   # [y, x, z]
    two_layer = two_layer.transpose(2, 0, 1)    # [z, y, x]
    resize_x, resize_y = 1.0, float(z2 - z1 + 1) / 2
    interpolation = cv2.INTER_LINEAR

    if two_layer.shape[2] > 512:
        res1 = two_layer[..., :512]
        res2 = two_layer[..., 512:]
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res = np.dstack([res1, res2])
    else:
        res = cv2.resize(two_layer, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    
    res[res > 0.5] = 1
    res[res <= 0.5] = 0

    return res


def generate_3d_cubes(final_spacing, cube_size, log=True, dump_img=False, process_only_patient=None):
    nodule_paths = glob(os.path.join(allPath.SA_SEG_DIR, "*.txt"))
    process_only_patient_flag = True

    if nodule_paths == []:
        return False

    if not os.path.exists(allPath.SA_SEG_CUBE_DIR):
        os.mkdir(allPath.SA_SEG_CUBE_DIR)

    if process_only_patient is None:
        basepath = allPath.SA_SEG_CUBE_DIR
    else:
        basepath = allPath.SA_TEMP_DIR
        if not os.path.exists(allPath.SA_TEMP_DIR):
            os.mkdir(allPath.SA_TEMP_DIR)

    last_patient_id = ''
    for i, nodule_path in enumerate(nodule_paths):

        base_name = os.path.basename(nodule_path)[:-4]
        cube_path = os.path.join(basepath, base_name)

        patient_id = base_name[2:66]
        nodule_No = base_name[67:]

        if process_only_patient is not None and process_only_patient != patient_id:
            continue

        if log:
            print(i, patient_id, nodule_No)

        if patient_id != last_patient_id:
            src_path = find_mhd_file(patient_id)
            if src_path is None:
                print("Error: Not belong to Luna16")
                continue

            # Read source image
            itk_image = SimpleITK.ReadImage(src_path)
            img_array = SimpleITK.GetArrayFromImage(itk_image)
            spacing = np.array(itk_image.GetSpacing())
            shape = img_array.shape     # channels, height, width
            old_shape = np.array([shape[2], shape[1], shape[0]])

            # Resample
            img_array = rescale_patient_images(img_array, spacing, final_spacing)   # channels, height, width
            img_array = normalize(img_array) * 255              # Normalize to 0-255
            shape = img_array.shape
            new_shape = np.array([shape[2], shape[1], shape[0]])

            if dump_img:
                if not tf.gfile.Exists(allPath.SA_IMGS_DIR):
                    tf.gfile.MkDir(allPath.SA_IMGS_DIR)

                patient_dir = os.path.join(allPath.SA_IMGS_DIR, patient_id)
                if not tf.gfile.Exists(patient_dir):
                    tf.gfile.MkDir(patient_dir)
                    
                for i in range(img_array.shape[0]):
                    img = img_array[i]
                    cv2.imwrite(os.path.join(patient_dir, str(i) + ".png"), img)


        with open(nodule_path, "r") as f:
            points = f.readlines()
        points = np.array([np.array((point.strip())[1:-1].split(', '), dtype=np.int) for point in points])
        # Calculate new coordinates
        new_points = points / old_shape * new_shape
        # Calculate center of the segmentated area
        center = (np.max(new_points, axis=0) + np.min(new_points, axis=0)) / 2
        cube_img, start_point = get_cube_from_image(img_array, center, cube_size)

        if cube_img.sum() < 5:
            print(" ***** Skipping ", center[0], center[1], center[2])
            continue

        if cube_img.mean() < 10:
            print(" ***** Suspicious ", center[0], center[1], center[2])
        
        if cube_img.shape != (64, 64, 64):
            print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
            continue
        
        save_cube_img(cube_path + "_o.png", cube_img, 8, 8)

        # Get nodule mask
        img_mask = np.zeros(old_shape[::-1], dtype=np.float)     # [z, y, x]
        img_mask[np.fliplr(points).T.tolist()] = 1
        # Fill holes
        sorted_z = np.sort(np.unique(points[:,2]))
        for i in sorted_z:
            img_mask[i] = roberts(img_mask[i])
            img_mask[i] = binary_fill_holes(img_mask[i])
            selem = disk(3)
            img_mask[i] = binary_erosion(img_mask[i], selem=selem)
        # # Interpolation along z axis
        # for i in range(len(sorted_z) - 1):
        #     if sorted_z[i+1] - sorted_z[i] > 1:
        #         img_mask[sorted_z[i]:sorted_z[i+1] + 1] = z_interpolation(img_mask, sorted_z[i], sorted_z[i+1])
        img_mask = rescale_patient_images(img_mask, spacing, final_spacing)   # channels, height, width
        img_mask[img_mask > 0.5] = 1
        img_mask[img_mask < 0.5] = 0
        
        cube_mask, start_point = get_cube_from_image(img_mask * 255, center, cube_size)

        save_cube_img(cube_path + "_m.png", cube_mask, 8, 8)

        # Release memory
        del(img_mask)
        del(cube_mask)

        if process_only_patient is not None and \
            last_patient_id != patient_id and \
            not process_only_patient_flag:
            break
        
        if process_only_patient is not None and \
            last_patient_id != patient_id and \
            process_only_patient_flag:
            process_only_patient_flag = False

        last_patient_id = patient_id
    
    return True


def random_divide_dataset():
    """ Divide Luna16 dataset (1572) into three part:
    * training set      (1200)
    * validation set    (150)
    * test set          (222)
    """
    train_num = 1200
    validation_num = 150
    test_set = 222
    
    cube_dir = allPath.SA_SEG_CUBE_DIR
    all_files = glob(os.path.join(cube_dir, "*o.png"))
    np.random.shuffle(all_files)


    with open(os.path.join(allPath.SA_ROOT_DIR, "train.csv"), "w") as f:
        f.write('original image, mask\n')
        for one_file in all_files[:train_num]:
            mask = one_file[:-5] + 'm.png'
            f.write(one_file + ',' + mask + '\n')
    with open(os.path.join(allPath.SA_ROOT_DIR, "validation.csv"), "w") as f:
        f.write('original image, mask\n')
        for one_file in all_files[train_num:train_num+validation_num]:
            mask = one_file[:-5] + 'm.png'
            f.write(one_file + ',' + mask + '\n')
    with open(os.path.join(allPath.SA_ROOT_DIR, "test.csv"), "w") as f:
        f.write('original image, mask\n')
        for one_file in all_files[train_num+validation_num:]:
            mask = one_file[:-5] + 'm.png'
            f.write(one_file + ',' + mask + '\n')
    


if __name__ == '__main__':
    process_only_patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.214252223927572015414741039150"
    process_only_patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"
    process_only_patient = "1.3.6.1.4.1.14519.5.2.1.6279.6001.168833925301530155818375859047"
    if False:
        process_lidc_segmentation(log=True, screening=True, 
                                dump_annos=True,
                                dump_chars=False, 
                                process_only_patient=None, 
                                begin_with_No=0)

    if False:
        process_lidc_segmentation(log=True, screening=True,
                                dump_annos=False,
                                dump_chars=True,
                                process_only_patient=None,
                                begin_with_No=0)

    if False:
        spacing = 0.7
        cube_size = 64
        generate_3d_cubes(final_spacing=spacing, cube_size=cube_size, dump_img=False, process_only_patient=None)

    if True:
        random_divide_dataset()

