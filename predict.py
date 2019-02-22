import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

import glob
import glog as log
import time

import matplotlib.pyplot as plt

from utils import utils, helpers
from builders import model_builder
import lanenet_postprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # BY TMM

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default= "/home/mmtao/test/lanenet-lane-detection/data/tusimple_test_image/b.jpg", required=False, help='The image you want to predict on. ')
#parser.add_argument('--image_dir', type=str, default= "/home/mmtao/test/lanenet-lane-detection/data/tusimple_test_image/05250538_0313.MP4", required=False, help='The image dir you want to predict on. ')
parser.add_argument('--image_dir', type=str, default= "/home/mmtao/anzhi-2018-09-22-08-51-51-089", required=False, help='The image dir you want to predict on. ')
parser.add_argument('--save_dir', type=str, default= "/home/mmtao/test/saveimg",help='Test result image save dir')
parser.add_argument('--checkpoint_path', type=str, default="/home/mmtao/test/Semantic-Segmentation-Suite/checkpoints/latest_model_BiSeNet_CamVid.ckpt", required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=576, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="BiSeNet", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[1,args.crop_height,args.crop_width,3])
net_output = tf.placeholder(tf.float32,shape=[1,args.crop_height,args.crop_width,num_classes])

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=tf.cast(False, tf.bool))

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

"""
print("Testing image " + args.image)

loaded_image = utils.load_image(args.image)
resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

st = time.time()
output_image = sess.run(network,feed_dict={net_input:input_image})

run_time = time.time()-st

output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

#out_vis_image = cv2.resize(out_vis_image, (1022,  573))
file_name = utils.filepath_to_name(args.image)
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
#cv2.imwrite("%s_pred11.png"%(file_name),cv2.cvtColor(np.uint8(output_image), cv2.COLOR_RGB2BGR))
"""

"""
image_dir = args.image_dir
image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)
for image in image_path_list:
"""

videoFileName = '/home/mmtao/data/2018-11-30_14-17-28.avi'
cap = cv2.VideoCapture(videoFileName)
if False == cap.isOpened():
    print("open video failed!")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/mmtao/data/output_%s' % os.path.basename(videoFileName),fourcc, 20, (args.crop_width, args.crop_height))

while(cap.isOpened()):
    ret, loaded_image = cap.read()
    #cv2.imshow('image', image)
    #k = cv2.waitKey(20)
    #loaded_image = utils.load_image(image)
#    if loaded_image==None:
#        continue
    resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))

    """
    if (args.crop_width <= loaded_image.shape[1]) and (args.crop_height <= loaded_image.shape[0]):
        x = np.int((loaded_image.shape[1] - args.crop_width) / 2)
        y = np.int((loaded_image.shape[0] - args.crop_height) / 2)
        resized_image=loaded_image[y:loaded_image.shape[0]-y, x:loaded_image.shape[1]-x, :]
    else:
        resized_image = loaded_image
    """

    input_image = np.expand_dims(np.float32(resized_image), axis=0) / 255.0

    st = time.time()
    output_image = sess.run(network, feed_dict={net_input: input_image})

    run_time = time.time() - st
    print(run_time)

    output_image = np.array(output_image[0, :, :, :])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    #unique_labels, unique_id, counts = tf.unique_with_counts(out_vis_image)

    #file_name = utils.filepath_to_name(image)

    #mask_image = cv2.resize(np.uint8(out_vis_image), (loaded_image.shape[1],
    #                                       loaded_image.shape[0]),interpolation=cv2.INTER_LINEAR)
    #b_result = cv2.imwrite("%s_pred.png" % (file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.IMREAD_COLOR))

    out_vis_image[out_vis_image[:, :, 0] == 1] = [255, 0, 0]
    out_vis_image[out_vis_image[:, :, 0] == 2] = [0, 255, 0]
    out_vis_image[out_vis_image[:, :, 0] == 3] = [0, 0, 255]
    out_vis_image[out_vis_image[:, :, 0] == 4] = [255, 255, 0]

    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
    postprocessor.postprocess(out_vis_image)

    resized_image = np.array(resized_image, np.uint8)
    out_vis_image = np.array(out_vis_image, np.uint8)
    add_image = cv2.addWeighted(resized_image, 1, out_vis_image, 1, 0)

    out.write(add_image)
    cv2.imshow('frame', add_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #b_result = cv2.imwrite("%s_pred.png" % (file_name), cv2.cvtColor(add_image, cv2.IMREAD_COLOR))
    #plt.ion()
    #plt.figure('add_image')
    #plt.imshow(add_image[:, :, (2, 1, 0)])
    #plt.show()
    #plt.pause(0.0000001)
    #plt.ioff()

    # cv2.imwrite("%s_pred11.png"%(file_name),cv2.cvtColor(np.uint8(output_image), cv2.COLOR_RGB2BGR))
cap.release()
out.release()
cv2.destroyAllWindows()
print("")
print("Finished!")
#print("Wrote image " + "%s_pred.png"%(file_name))
