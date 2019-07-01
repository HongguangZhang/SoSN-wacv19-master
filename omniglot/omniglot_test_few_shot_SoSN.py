#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random
import models

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num_per_class",type = int, default = 5)
parser.add_argument("-b","--query_num_per_class",type = int, default = 2)
parser.add_argument("-e","--episode",type = int, default= 100)
parser.add_argument("-t","--query_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-sigma","--sigma", type = float, default = 1)
parser.add_argument("-ts","--test_num_per_class",type = int, default = 5)
args = parser.parse_args()


# Hyper Parameters
METHOD = "SoSN_LOGIT" + str(args.sigma) + "_Models"
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.support_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
TEST_NUM_PER_CLASS = args.test_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.query_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma

def power_norm(x, SIGMA):
	out = 2/(1 + torch.exp(-SIGMA*x)) - 1
	return out
	
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metaquery_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    relation_network = models.SimilarityNetwork(FEATURE_DIM,RELATION_DIM).apply(weights_init).cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=50000,gamma=0.1)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=50000,gamma=0.1)

    if os.path.exists(str(METHOD + "/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(METHOD + "/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str(METHOD + "/omniglot_similarity_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str(METHOD + "/omniglot_similarity_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load similarity network success")
    if os.path.exists(METHOD) == False:
        os.system('mkdir ' + METHOD)

    # Step 3: build graph
    print("Training...")

    best_accuracy = 0.0
    best_h = 0.0

    for episode in range(EPISODE):
        with torch.no_grad():   
            # query
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                degrees = random.choice([0,90,180,270])
                task = tg.OmniglotTask(metaquery_character_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,TEST_NUM_PER_CLASS,)
                support_dataloader = tg.get_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
                query_dataloader = tg.get_data_loader(task,num_per_class=TEST_NUM_PER_CLASS,split="query",shuffle=True,rotation=degrees)

                support_images,support_labels = support_dataloader.__iter__().next()
                query_images,query_labels = query_dataloader.__iter__().next()
                
                # calculate features
                support_features = feature_encoder(Variable(support_images).cuda(GPU)) # 5x64
                support_features = support_features.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,25).sum(1)
                query_features = feature_encoder(Variable(query_images).cuda(GPU)).view(TEST_NUM_PER_CLASS*CLASS_NUM,64,25)

                H_support_features = Variable(torch.Tensor(CLASS_NUM, 1, 64, 64)).cuda(GPU)
                H_query_features = Variable(torch.Tensor(TEST_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda(GPU)
                # HOP features
                for d in range(support_features.size(0)):
                    s = support_features[d,:,:].squeeze(0)
                    s = (1.0 / support_features.size(2)) * s.mm(s.t())
                    H_support_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                for d in range(query_features.size(0)):
                    s = query_features[d,:,:].squeeze(0)
                    s = (1.0 / query_features.size(2)) * s.mm(s.t())
                    H_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                    
                # calculate relations
                # each query support link to every supports to calculate relations
                # to form a 100x128 matrix for relation network
                support_features_ext = H_support_features.unsqueeze(0).repeat(TEST_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                query_features_ext = H_query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                query_features_ext = torch.transpose(query_features_ext,0,1)

                relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
                relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(CLASS_NUM*TEST_NUM_PER_CLASS)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/TEST_NUM_PER_CLASS/TEST_EPISODE

            print("query accuracy:",test_accuracy)
            print("best accuracy:",best_accuracy)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy





if __name__ == '__main__':
    main()
