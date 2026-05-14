
import torch.nn as nn
import torch
from eda_strategies.UMDA import UMDA
from eda_strategies.PBIL import PBIL
from eda_strategies.PPO_EDA import PPO_EDA



class FactoryStrategyEA:

    def createStrategyEA(self, typeStrategy, N, lambda_, alpha, beta, device, typeModel, numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnDAG, noise_rescale, modeCritic, shareParameters, nb_train, coeff_dropout, mode_gibbs_sampling, nb_sampling_gibbs):

        if (typeStrategy == "UMDA"):
            return UMDA(N, lambda_, device)

        elif(typeStrategy == "PBIL"):

            return PBIL(N, lambda_, device)


        elif (typeStrategy == "PPO-EDA"):
            
            print("OK")

            return PPO_EDA(N,  lambda_, alpha, beta, device, typeModel,numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnDAG, noise_rescale, modeCritic, shareParameters, nb_train, coeff_dropout, mode_gibbs_sampling, nb_sampling_gibbs)
