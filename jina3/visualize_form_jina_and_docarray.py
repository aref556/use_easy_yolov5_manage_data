
import itertools
import pandas as pd
import numpy as np
import time

from openvino.inference_engine import IECore
import cv2 as cv

from sklearn.metrics.pairwise import cosine_distances as cos_distance
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity

import matplotlib.pyplot as plt

import os

# import sys
# sys.path.append("../utils")
# from notebook_utils import (
#     viz_result_image,
# )

#-------------------  CONFIGURE -------------------
PATH_DIRECTORY_DATA = 'data/persons_another_view'
PATH_DIRECTORY_MODEL = 'models/FP32'




identities = {
    "person01": [
        "p1-1.jpg", "p1-2.jpg", "p1-3.jpg", "p1-4.jpg", "p1-5.jpg", "p1-6.jpg", "p1-7.jpg", "p1-8.jpg", "p1-9.jpg", "p1-10.jpg", 
        "p1-11.jpg", "p1-12.jpg", "p1-13.jpg", "p1-14.jpg", "p1-15.jpg", "p1-16.jpg", "p1-17.jpg", "p1-18.jpg", "p1-19.jpg", "p1-20.jpg",
        "p1-21.jpg", "p1-22.jpg", "p1-23.jpg", "p1-24.jpg", "p1-25.jpg", "p1-26.jpg", "p1-27.jpg", "p1-28.jpg", "p1-29.jpg", "p1-30.jpg"
    ],
    "person02": [
        "p2-1.jpg", "p2-2.jpg", "p2-3.jpg", "p2-4.jpg", "p2-5.jpg", "p2-6.jpg", "p2-7.jpg", "p2-8.jpg", "p2-9.jpg", "p2-10.jpg",
        "p2-11.jpg", "p2-12.jpg", "p2-13.jpg", "p2-14.jpg", "p2-15.jpg", "p2-16.jpg", "p2-17.jpg", "p2-18.jpg", "p2-19.jpg", "p2-20.jpg",
        "p2-21.jpg", "p2-22.jpg", "p2-23.jpg", "p2-24.jpg", "p2-25.jpg", "p2-26.jpg", "p2-27.jpg", "p2-28.jpg", "p2-29.jpg", "p2-30.jpg"
    ],
    "person03": [
        "p3-1.jpg", "p3-2.jpg", "p3-3.jpg", "p3-4.jpg", "p3-5.jpg", "p3-6.jpg", "p3-7.jpg", "p3-8.jpg", "p3-9.jpg", "p3-10.jpg",
        "p3-11.jpg", "p3-12.jpg", "p3-13.jpg", "p3-14.jpg", "p3-15.jpg", "p3-16.jpg", "p3-17.jpg", "p3-18.jpg", "p3-19.jpg", "p3-20.jpg",
        "p3-21.jpg", "p3-22.jpg", "p3-23.jpg", "p3-24.jpg", "p3-25.jpg", "p3-26.jpg", "p3-27.jpg", "p3-28.jpg", "p3-29.jpg", "p3-30.jpg"
    ],
    "person04": [
        "p4-1.jpg", "p4-2.jpg", "p4-3.jpg", "p4-4.jpg", "p4-5.jpg", "p4-6.jpg", "p4-7.jpg", "p4-8.jpg", "p4-9.jpg", "p4-10.jpg",
        "p4-11.jpg", "p4-12.jpg", "p4-13.jpg", "p4-14.jpg", "p4-15.jpg", "p4-16.jpg", "p4-17.jpg", "p4-18.jpg", "p4-19.jpg", "p4-20.jpg",
        "p4-21.jpg", "p4-22.jpg", "p4-23.jpg", "p4-24.jpg", "p4-25.jpg", "p4-26.jpg", "p4-27.jpg", "p4-28.jpg", "p4-29.jpg", "p4-30.jpg"
    ],
    "person05": [
        "p5-1.jpg", "p5-2.jpg", "p5-3.jpg", "p5-4.jpg", "p5-5.jpg", "p5-6.jpg", "p5-7.jpg", "p5-8.jpg", "p5-9.jpg", "p5-10.jpg",
        "p5-11.jpg", "p5-12.jpg", "p5-13.jpg", "p5-14.jpg", "p5-15.jpg", "p5-16.jpg", "p5-17.jpg", "p5-18.jpg", "p5-19.jpg", "p5-20.jpg",
        "p5-21.jpg", "p5-22.jpg", "p5-23.jpg", "p5-24.jpg", "p5-25.jpg", "p5-26.jpg", "p5-27.jpg", "p5-28.jpg", "p5-29.jpg", "p5-30.jpg"
    ],
    "person06": [
        "p6-1.jpg", "p6-2.jpg", "p6-3.jpg", "p6-4.jpg", "p6-5.jpg", "p6-6.jpg", "p6-7.jpg", "p6-8.jpg", "p6-9.jpg", "p6-10.jpg",
        "p6-11.jpg", "p6-12.jpg", "p6-13.jpg", "p6-14.jpg", "p6-15.jpg", "p6-16.jpg", "p6-17.jpg", "p6-18.jpg", "p6-19.jpg", "p6-20.jpg",
        "p6-21.jpg", "p6-22.jpg", "p6-23.jpg", "p6-24.jpg", "p6-25.jpg", "p6-26.jpg", "p6-27.jpg", "p6-28.jpg", "p6-29.jpg", "p6-30.jpg"
    ],
    "person07": [
        "p7-1.jpg", "p7-2.jpg", "p7-3.jpg", "p7-4.jpg", "p7-5.jpg", "p7-6.jpg", "p7-7.jpg", "p7-8.jpg", "p7-9.jpg", "p7-10.jpg",
        "p7-11.jpg", "p7-12.jpg", "p7-13.jpg", "p7-14.jpg", "p7-15.jpg", "p7-16.jpg", "p7-17.jpg", "p7-18.jpg", "p7-19.jpg", "p7-20.jpg",
        "p7-21.jpg", "p7-22.jpg", "p7-23.jpg", "p7-24.jpg", "p7-25.jpg", "p7-26.jpg", "p7-27.jpg", "p7-28.jpg", "p7-29.jpg", "p7-30.jpg"
    ],
    "person08": [
        "p8-1.jpg", "p8-2.jpg", "p8-3.jpg", "p8-4.jpg", "p8-5.jpg", "p8-6.jpg", "p8-7.jpg", "p8-8.jpg", "p8-9.jpg", "p8-10.jpg",
        "p8-11.jpg", "p8-12.jpg", "p8-13.jpg", "p8-14.jpg", "p8-15.jpg", "p8-16.jpg", "p8-17.jpg", "p8-18.jpg", "p8-19.jpg", "p8-20.jpg",
        "p8-21.jpg", "p8-22.jpg", "p8-23.jpg", "p8-24.jpg", "p8-25.jpg", "p8-26.jpg", "p8-27.jpg", "p8-28.jpg", "p8-29.jpg", "p8-30.jpg"
    ],
    "person09": [
        "p9-1.jpg", "p9-2.jpg", "p9-3.jpg", "p9-4.jpg", "p9-5.jpg", "p9-6.jpg", "p9-7.jpg", "p9-8.jpg", "p9-9.jpg", "p9-10.jpg",
        "p9-11.jpg", "p9-12.jpg", "p9-13.jpg", "p9-14.jpg", "p9-15.jpg", "p9-16.jpg", "p9-17.jpg", "p9-18.jpg", "p9-19.jpg", "p9-20.jpg",
        "p9-21.jpg", "p9-22.jpg", "p9-23.jpg", "p9-24.jpg", "p9-25.jpg", "p9-26.jpg", "p9-27.jpg", "p9-28.jpg", "p9-29.jpg", "p9-30.jpg"
    ],
    "person10": [
        "p10-1.jpg", "p10-2.jpg", "p10-3.jpg", "p10-4.jpg", "p10-5.jpg", "p10-6.jpg", "p10-7.jpg", "p10-8.jpg", "p10-9.jpg", "p10-10.jpg",
        "p10-11.jpg", "p10-12.jpg", "p10-13.jpg", "p10-14.jpg", "p10-15.jpg", "p10-16.jpg", "p10-17.jpg", "p10-18.jpg", "p10-19.jpg", "p10-20.jpg",
        "p10-21.jpg", "p10-22.jpg", "p10-23.jpg", "p10-24.jpg", "p10-25.jpg", "p10-26.jpg", "p10-27.jpg", "p10-28.jpg", "p10-29.jpg", "p10-30.jpg"
    ],
}

models = ["person-reidentification-retail-0277.xml", "person-reidentification-retail-0286.xml", "person-reidentification-retail-0287.xml", "person-reidentification-retail-0288.xml"]
metrics = ['cosine', 'similarity']

#--------------------------------------------------


positives = []
for key, values in identities.items() :
    print("key : ", key)
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            positive = []
            positive.append(values[i])
            positive.append(values[j])
            positives.append(positive)
positives = pd.DataFrame(positives, columns= ["file_x", "file_y"])
positives["decision"] = "Yes"

print(positives)

samples_list = list(identities.values())

negatives = []

for i in range(0, len(identities) - 1):
    for j in range(i+1, len(identities)):
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
        
        for cross_sample in cross_product:
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)
            
negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"
negatives = negatives.sample(positives.shape[0])
print(negatives)

df = pd.concat([positives, negatives]).reset_index(drop = True)

print(df)

instances = df[["file_x", "file_y"]].values.tolist()



# -------------------------- convert to vector ----------------------
time_models = {}
vector_dict = {}
list_data_name = os.listdir(PATH_DIRECTORY_DATA)

print('vector encode start')

for model in models:
    print("Model: {}".format(model))
    ir_path = "{}/{}".format(PATH_DIRECTORY_MODEL, model)
    
    # Load the network in Inference Engine
    ie = IECore()
    net_ir = ie.read_network(model=ir_path)
    exec_net_ir = ie.load_network(network=net_ir, device_name="CPU")
    input_layer = next(iter(net_ir.input_info))
    output_layer = next(iter(net_ir.outputs))

    time_model = []
    result_model = {}

    for i, namefile in enumerate(list_data_name):
        img_path = '{}/{}'.format(PATH_DIRECTORY_DATA, namefile)
        image = cv.imread(img_path)
        #  The net expects one input image of the shape 1, 3, 256, 128 in the B, C, H, W format, where:
        # B - batch size
        # C - number of channels
        # H - image height
        # W - image width

        # B,C,H,W = batch size, number of channels, height, width
        B, C, H, W = net_ir.input_info[input_layer].tensor_desc.dims

        # print("H, W : ", H, W)
        
        # OpenCV resize expects the destination size as (width, height)
        resized_image = cv.resize(src=image, dsize=(W, H))
        # print("resized_image : ", resized_image.shape)

        #  transpose is change center of array
        input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        
        time_start = time.time()

        result = exec_net_ir.infer({input_layer: input_data})

        time_end = time.time() - time_start
        time_model.append(time_end)

        vector_output = result[output_layer]
        result_model.update({namefile: vector_output})

    time_avg = sum(time_model) / len(time_model)
    # print("{} : {}".format(model, time_avg))
    time_models.update({model : time_avg})
    vector_dict.update({model: result_model})



print('vector encode finish')
print("------- show performance run instances --------")
# print('results : {}'.format(vector_dict))
print('model use time : {}'.format(time_models))

# -------------------------------------------------------------------


for model in models:
    print("Model: ", model)
    

    for metric in metrics:
        print("Metric: ", metric) 
        # print("inst: ", instances[0])
        
        
        distances = []
        
        result_model = vector_dict[model]
        print('result_model : {}'.format(result_model))
        for img_pair in instances:
            

            vector_1 = result_model[img_pair[0]]
            vector_2 = result_model[img_pair[1]]
           
            # print(output.shape)
            # print(output_2.shape)
            mid_dist = None
            if(metric == 'cosine'):
                distance = cos_distance(vector_1, vector_2)
                mid_dist = distance[0][0]
                # print("distance 2 vectors is : {}".format(distance))
            elif(metric == 'similarity'):
                similarity = cos_similarity(vector_1, vector_2, dense_output=True)
                mid_dist = similarity[0][0]
                # print("similarity 2 vectors is : {}". format(similarity))
            
            distances.append(mid_dist)

        df['%s_%s' % (model[:len(model)-4], metric)] = distances
        
print(df)


