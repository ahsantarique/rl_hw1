import numpy as np
from scipy.stats import bernoulli

class PHEStruct:
    def __init__(self, num_arm, c, a, p):
        self.d = num_arm

        self.UserArmMean = np.ones(self.d)
        self.UserArmTrials = np.ones(self.d)

        self.B = np.zeros(self.d) 

        self.a = int(a) ## hyperparam
        self.p = p
        self.c = c ## hyperparameter

        self.time = 0


    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        
        self.UserArmTrials[articlePicked_id] += 1


        self.time += 1 ## this comes before the next line so that we never encounter log 0

        self.B[articlePicked_id] = np.sqrt(2*np.log(self.time) / self.UserArmTrials[articlePicked_id])


    def getTheta(self):
        return self.UserArmMean

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

            noise = np.sum([bernoulli.rvs(self.p) for i in range(int(self.a*self.UserArmTrials[article.id]))])

            article_pta = ((self.UserArmMean[article.id] * self.UserArmTrials[article.id] + noise) / ((self.a+1) * self.UserArmTrials[article.id])) + self.c*self.B[article.id]
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked



class PHE:
    def __init__(self, num_arm,  c, a, p=0.5):
        self.users = {}
        self.num_arm = num_arm
        self.a = int(a) ## hyperparam
        self.p = p ## hyperparam
        self.c = c ## hyperparameter
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm, c= self.c, a = self.a, p=self.p)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID):
        return self.users[userID].UserArmMean


