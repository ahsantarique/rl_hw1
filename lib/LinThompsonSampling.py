import numpy as np
from scipy.stats import multivariate_normal


class UCBStruct:
    def __init__(self, num_arm, lambda_, var):
        self.d = num_arm
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_

        self.var = var


        self.b = np.zeros(self.d)

        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        # self.UserTheta = np.dot(self.AInv, self.b)

        # self.UserArmMean = np.zeros(self.d)
        # self.UserArmTrials = np.zeros(self.d)

        self.time = 0


    def updateParameters(self, articlePicked_id, click):

        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click

        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1 ## this comes before the next line so that we never encounter log 0


        # update a, b here

        ###################################


    def getTheta(self):
        return self.UserTheta



    def decide(self, pool_articles):
        if (self.time < self.d):
            t = 0
            for article in pool_articles:
                t += 1
                if (t > self.time):
                    return article

        maxPTA = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_pta = multivariate_normal.rvs(np.matmul(self.AInv, self.b), self.AInv)
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta
        return articlePicked



class UpperConfidenceBound:
    def __init__(self, num_arm, c):
        self.users = {}
        self.num_arm = num_arm
        self.c = c
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm, c= self.c)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean


