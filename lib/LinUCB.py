import numpy as np

class LinUCBStruct:
    def __init__(self, featureDimension, lambda_, c):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.lambda_ = lambda_

        self.b = np.zeros(self.d)


        self.c = c

        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d)
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1
       



    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

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
            
            confidence = np.matmul( (np.matmul(article.featureVector.T, self.AInv) ), article.featureVector)
            article_pta = np.dot(self.UserTheta, article.featureVector) + self.c*confidence
            # pick article with highest Prob
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class LinUCB:
    def __init__(self, dimension, lambda_, c):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.c = c

        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = LinUCBStruct(self.dimension, self.lambda_, self.c)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


