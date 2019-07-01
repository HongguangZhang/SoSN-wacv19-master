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
import scipy as sp
import scipy.stats
import models

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 2)
parser.add_argument("-e","--episode",type = int, default= 50000)
parser.add_argument("-t","--query_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-sigma","--sigma", type = float, default = 1)
parser.add_argument("-lam","--lamb", type = float, default = 0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "SoSN_Logit" + str(args.sigma) + "_Models"
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = 1
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.query_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma
LAMBDA = args.lamb

def power_norm(x, SIGMA):
	out = 2/(1 + torch.exp(-SIGMA*x)) - 1
	return out
	
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

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
    metatrain_folders,metaquery_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = models.FeatureEncoder().apply(weights_init).cuda(GPU)
    relation_network = models.SimilarityNetwork(FEATURE_DIM,RELATION_DIM).apply(weights_init).cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=50000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=50000,gamma=0.5)

    if os.path.exists(str(METHOD + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(METHOD + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str(METHOD + "/relation_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str(METHOD + "/relation_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    if os.path.exists(METHOD) == False:
        os.system('mkdir ' + METHOD)

    # Step 3: build graph
    print("Training...")

    best_accuracy = 0.0
    best_h = 0.0
            
    for episode in range(EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # support_dataloader is to obtain previous supports for compare
        # query_dataloader is to query supports for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=QUERY_NUM_PER_CLASS,split="test",shuffle=True)

        # support datas
        supports,support_labels = support_dataloader.__iter__().next()
        queries,query_labels = query_dataloader.__iter__().next()
        
        # calculate features
        support_features = feature_encoder(Variable(supports).cuda(GPU)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,64,19**2).sum(1) # 5x64*19*19
        query_features = feature_encoder(Variable(queries).cuda(GPU)).view(QUERY_NUM_PER_CLASS*CLASS_NUM,64,19**2) # 20x64*19*19
        H_support_features = Variable(torch.Tensor(SUPPORT_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda(GPU)
        H_query_features = Variable(torch.Tensor(QUERY_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda(GPU)
        # HOP features
        for d in range(support_features.size()[0]):
            s = support_features[d,:,:].squeeze(0)                        
            s = s - LAMBDA * s.mean(1).repeat(1,s.size()[1]).view(s.size())
            s = (1 / support_features.size()[2]) * s.mm(s.transpose(0,1))
            H_support_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
        for d in range(query_features.size()[0]):
            s = query_features[d,:,:].squeeze(0)
            s = s - LAMBDA * s.mean(1).repeat(1,s.size()[1]).view(s.size())
            s = (1 / query_features.size()[2]) * s.mm(s.transpose(0,1))
            H_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)

        # form relation pairs
        support_features_ext = H_support_features.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        query_features_ext = H_query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        query_features_ext = torch.transpose(query_features_ext,0,1)
        relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
        # calculate relation scores
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SUPPORT_NUM_PER_CLASS)

        # define loss function
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, query_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)

        # updating network parameters with their gradients
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if episode%500 == 0:
            # query
            print("Testing...")
            
            accuracies = []
            for i in range(TEST_EPISODE):
              with torch.no_grad():
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metaquery_folders,CLASS_NUM,1,2)
                support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)
                num_per_class = 2
                query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="query",shuffle=True)
                support_images,support_labels = support_dataloader.__iter__().next()
                for query_images,query_labels in query_dataloader:
                    query_size = query_labels.shape[0]
                    # calculate features
                    support_features = feature_encoder(Variable(support_images).cuda(GPU)).view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,64,19**2).sum(1) 
                    query_features = feature_encoder(Variable(query_images).cuda(GPU)).view(num_per_class*CLASS_NUM,64,19**2) 
                    
                    H_support_features = Variable(torch.Tensor(SUPPORT_NUM_PER_CLASS*CLASS_NUM, 1, 64, 64)).cuda(GPU)
                    H_query_features = Variable(torch.Tensor(num_per_class*CLASS_NUM, 1, 64, 64)).cuda(GPU)
                    # HOP features
                    for d in range(support_features.size()[0]):
                        s = support_features[d,:,:].squeeze(0)
                        s = s - LAMBDA * s.mean(1).repeat(1,s.size()[1]).view(s.size())
                        s = (1 / support_features.size()[2]) * s.mm(s.transpose(0,1))
                        H_support_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)
                    for d in range(query_features.size()[0]):
                        s = query_features[d,:,:].squeeze(0)
                        s = s - LAMBDA * s.mean(1).repeat(1,s.size()[1]).view(s.size())
                        s = (1 / query_features.size()[2]) * s.mm(s.transpose(0,1))
                        H_query_features[d,:,:,:] = power_norm(s / s.trace(), SIGMA)

                    # form relation pairs
                    support_features_ext = H_support_features.unsqueeze(0).repeat(query_size,1,1,1,1)
                    query_features_ext = H_query_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    query_features_ext = torch.transpose(query_features_ext,0,1)
                    relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,2,64,64)
                    # calculate relation scores
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(query_size)]

                    total_rewards += np.sum(rewards)
                    counter += query_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("Test accuracy:", test_accuracy,"h:",h)
            print("Best accuracy: ", best_accuracy, "h:", best_h)

            if test_accuracy > best_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(),str(METHOD + "/feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str(METHOD + "/relation_network_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode)

                best_accuracy = test_accuracy
                best_h = h





if __name__ == '__main__':
    main()
