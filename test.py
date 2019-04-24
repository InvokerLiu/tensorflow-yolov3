#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
from PIL import Image
sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 1
SHUFFLE_SIZE = 200
CLASSES = utils.read_coco_names("data/SVHN/SVHN.names")
ANCHORS = utils.get_anchors("data/SVHN/SVHN_anchors.txt", IMAGE_H, IMAGE_W)
NUM_CLASSES = len(CLASSES)
test_tfrecord = "data/SVHN/tfrecords/quick_test_data.tfrecords"

parser = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
testset = dataset(parser, test_tfrecord, BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = testset.get_next()

images, *y_true = example
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)

with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(images, is_training=is_training)
    loss = model.compute_loss(pred_feature_map, y_true)
    y_pred = model.predict(pred_feature_map)

saver = tf.train.Saver()
saver.restore(sess, "data/SVHN/checkpoint5/yolov3.ckpt-4000")

acc = 0
STEPS = 13068

for step in range(STEPS):
    run_items = sess.run([y_pred, y_true], feed_dict={is_training: False})
    if step == 5:
        acc = utils.compute_accuracy(run_items[0], run_items[1])

    y_pred_data = run_items[0]
    pred_boxes = y_pred_data[0][0]
    pred_confs = y_pred_data[1][0]
    pred_probs = y_pred_data[2][0]

    pred_boxes, pred_scores, pred_labels = utils.cpu_nms(pred_boxes, pred_confs * pred_probs, NUM_CLASSES,
                                                         score_thresh=0.3, iou_thresh=0.5)

    img = Image.open("data/SVHN/PaddingTest/" + str(step + 1) + ".png")
    image = utils.draw_boxes(img, pred_boxes, pred_scores, pred_labels, CLASSES, [IMAGE_H, IMAGE_W], show=False)
    # if acc == 1:
    #     image.save("data/SVHN/RightRecognition/" + str(step + 1) + ".png")
    # else:
    #     image.save("data/SVHN/WrongRecognition/" + str(step + 1) + ".png")
    print("=> STEP %10d [VALID]:\tacc:%7.4f" % (step+1, acc))
    acc += acc * BATCH_SIZE

acc /= 13068
print("精度为%7.4f" % acc)
