# -*-coding:utf-8-*-
"""
Author: Bo Liu
"""
import h5py
import sys
import argparse
from PIL import Image
import numpy


def get_box_data(index, hdf5_data):
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def read_mat(file_name, begin, end):
    f = h5py.File(file_name, 'r')
    names = list()
    boxes = list()
    for i in range(begin, end):
        pic = get_name(i, f)
        names.append(pic)
        box = get_box_data(i, f)
        boxes.append(box)
    f.close()
    return names, boxes


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default='../data/SVHN/test')
    parser.add_argument("--dataset_txt",
                        default='../data/SVHN/quick_test_data.txt')
    flags = parser.parse_args()
    digitStruct = flags.dataset_dir + "/digitStruct.mat"
    dataset = h5py.File(digitStruct, 'r')
    name = dataset['digitStruct']['name']
    data_size = name.size
    with open(flags.dataset_txt, "w") as f:
        names, boxes = read_mat(digitStruct, 0, data_size)
        for i in range(data_size):
            name = flags.dataset_dir + "/" + names[i]
            class_ids = boxes[i]["label"]

            image = Image.open(name)
            image_width, image_height = image.size[0], image.size[1]
            image = numpy.array(image)
            image_pad = image
            if image_width > image_height:
                padding_size = (image_width - image_height) // 2
                image_pad = numpy.pad(image, ((padding_size, padding_size), (0, 0), (0, 0)), "constant")
            if image_height > image_width:
                padding_size = (image_height - image_width) // 2
                image_pad = numpy.pad(image, ((0, 0), (padding_size, padding_size), (0, 0)), "constant")
            image = Image.fromarray(image_pad)
            file_name = "../data/SVHN/PaddingTest/" + str(i + 1) + ".png"
            image.save(file_name)
            f.write(file_name + " ")

            for j in range(len(class_ids)):
                left = boxes[i]["left"][j]
                top = boxes[i]["top"][j]
                if image_width > image_height:
                    top += padding_size
                if image_height > image_width:
                    left += padding_size
                f.write(str(left) + " ")
                f.write(str(top) + " ")
                f.write(str(left + boxes[i]["width"][j]) + " ")
                f.write(str(top + boxes[i]["height"][j]) + " ")
                if j == len(class_ids) - 1:
                    f.write(str(int(class_ids[j]) - 1) + "\n")
                else:
                    f.write(str(int(class_ids[j]) - 1) + " ")


if __name__ == "__main__":
    main(sys.argv[1:])
