import numpy as np
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
    def __init__(self, id, theta = None, CoTheta = None):
        self.id = id
        self.theta = theta

class UserManager():
    def __init__(self, dimension, userNum, thetaFunc, gamma=None, UserGroups=1, epsilon=None, argv = None):
        self.dimension = dimension
        self.thetaFunc = thetaFunc
        self.userNum = userNum
        self.gamma = gamma
        self.epsilon = epsilon
        self.UserGroups = UserGroups
        self.argv = argv
        self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

    def generateMasks(self):
        mask = {}
        for i in range(self.UserGroups):
            mask[i] = np.random.randint(2, size = self.dimension)
        return mask


    def simulateThetaForHomoUsers(self):
        users = []
        thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
        l2_norm = np.linalg.norm(thetaVector, ord=2)
        thetaVector = thetaVector/l2_norm
        for key in range(self.userNum):
            users.append(User(key, thetaVector))

        return users

    def simulateThetaForHeteroUsers(self, global_dim):
        local_dim = self.dimension-global_dim
        users = []
        thetaVector_g = self.thetaFunc(global_dim, argv=self.argv)
        l2_norm = np.linalg.norm(thetaVector_g, ord=2)
        thetaVector_g = thetaVector_g/l2_norm
        for key in range(self.userNum):
            thetaVector_l = self.thetaFunc(local_dim, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector_l, ord=2)
            thetaVector_l = thetaVector_l/l2_norm

            thetaVector = np.concatenate([thetaVector_g, thetaVector_l])
            users.append(User(key, thetaVector))

        return users

    def simulateThetaForClusteredUsers(self):
        users = []
        # Generate a global unique parameter set
        global_parameter_set = []
        for i in range(self.UserGroups):
            thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector, ord=2)
            new_theta = thetaVector / l2_norm

            if global_parameter_set == []:
                global_parameter_set.append(new_theta)
            else:
                dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in global_parameter_set])
                while (not dist_to_all_existing_big):
                    thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    new_theta = thetaVector / l2_norm
                    dist_to_all_existing_big = all(
                        [np.linalg.norm(new_theta - existing_theta) >= self.gamma for existing_theta in
                         global_parameter_set])
                global_parameter_set.append(new_theta)
        global_parameter_set = np.array(global_parameter_set)
        assert global_parameter_set.shape == (self.UserGroups, self.dimension)
        # Uniformly sample a parameter for each user as initial parameter
        parameter_index_for_users = np.random.randint(self.UserGroups, size=self.userNum)
        print(parameter_index_for_users)

        for key in range(self.userNum):
            parameter_index = parameter_index_for_users[key]
            users.append(User(key, global_parameter_set[parameter_index]))
            assert users[key].id == key
            assert np.linalg.norm(global_parameter_set[parameter_index] - users[key].theta) <= 0.001

        return users, global_parameter_set, parameter_index_for_users
    
    def simulateThetaForLooselyClusteredUsers(self):
        users = []
        # Generate a global unique parameter set for the cluster centers
        cluster_centers = []
        for i in range(self.UserGroups):
            thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
            l2_norm = np.linalg.norm(thetaVector, ord=2)
            new_theta = thetaVector / l2_norm

            if cluster_centers == []:
                cluster_centers.append(new_theta)
            else:
                dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= self.gamma+2*self.epsilon for existing_theta in cluster_centers])
                while (not dist_to_all_existing_big):
                    thetaVector = self.thetaFunc(self.dimension, argv=self.argv)
                    l2_norm = np.linalg.norm(thetaVector, ord=2)
                    new_theta = thetaVector / l2_norm
                    dist_to_all_existing_big = all(
                        [np.linalg.norm(new_theta - existing_theta) >= self.gamma+2*self.epsilon for existing_theta in
                         cluster_centers])
                cluster_centers.append(new_theta)
        cluster_centers = np.array(cluster_centers)
        assert cluster_centers.shape == (self.UserGroups, self.dimension)

        # Uniformly sample a cluster center for each user as initial parameter
        parameter_index_for_users = np.random.randint(self.UserGroups, size=self.userNum)
        print(parameter_index_for_users)

        #import pdb; pdb.set_trace()

        # sample within epsilon for each cluster center
        for key in range(self.userNum): #iterate over the users
            parameter_index = parameter_index_for_users[key]
            # compute a unique random vector to add to cluster centers for each user
            # such that it is no more than epsilon away from the centers
            stdev = np.identity(self.dimension) * (self.epsilon**3)
            user_theta = np.random.multivariate_normal(cluster_centers[parameter_index], stdev)

            # rejection sample to make sure we are within epsilon of the cluster center
            
            dist_cluster_center_small = np.linalg.norm(cluster_centers[parameter_index]-user_theta,ord=2)< self.epsilon
            while (not dist_cluster_center_small):
                 user_theta = np.random.multivariate_normal(cluster_centers[parameter_index], stdev)
                 #import pdb; pdb.set_trace()
                 dist_cluster_center_small = np.linalg.norm(cluster_centers[parameter_index]-user_theta,ord=2)< self.epsilon
                
            users.append(User(key, user_theta))
            assert users[key].id == key
            assert np.linalg.norm(cluster_centers[parameter_index] - users[key].theta,ord=2) < self.epsilon

        print('USER SAMPLING COMPLETE')
        return users, cluster_centers, parameter_index_for_users

