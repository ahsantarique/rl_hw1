import numpy as np

class UCBStruct:
    def __init__(self, num_arm, c):
        self.d = num_arm

        self.UserArmMean = np.zeros(self.d)
        self.UserArmTrials = np.zeros(self.d)

        self.B = np.zeros(self.d) 

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
            article_pta = self.UserArmMean[article.id] + self.c*self.B[article.id]
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


