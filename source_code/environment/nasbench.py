import torch
from tqdm import tqdm
#from bbdob.utils import idx2one_hot
import numpy as np


def  get_Score_trajectories_nasbench_cuda(objective, strategy, nb_instances, budget,  size_popEA, device, verbose, name_file):


    strategy.reset_learned_parameters(nb_instances)

    bestScore = torch.ones(nb_instances).to(device)*(-99999)


    size_pop = strategy.lambda_


    nb_iterations = budget//size_pop


    if(verbose):
        pbar = tqdm(range(nb_iterations))
    else:
        pbar = range(nb_iterations)


    if(name_file is not None):
        f_results = open(name_file, "w")
        f_results.write("runtime, mean, median, std, 2%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 98%" + "\n")
        f_results.close()

    for epoch in pbar:

        tensor_solution = strategy.sample_solutions()

        if(epoch == 0):
            print("tensor_solution.size()")
            print(tensor_solution.size())
            print(tensor_solution[0][0])

        tensor_solution_new = torch.nn.functional.one_hot(tensor_solution.squeeze(3).long(),-1).float()

        # print("tensor_solution_new.size()")
        # print(tensor_solution_new.size())
        # print(tensor_solution_new[0][0])

        tensor_solution_cpu = tensor_solution_new.squeeze().cpu().numpy()



        tensor_score = torch.zeros((nb_instances, size_popEA))

        for i in range(nb_instances):
            for j in range(size_popEA):

                solution = tensor_solution_cpu[i, j].astype(np.int)

                # print(solution)

                evals, info = objective(solution)



                tensor_score[i, j] = -evals[0]


        if(epoch == 0):
            print("tensor_score.size()")
            print(tensor_score.size())
            print(tensor_score[0][0])
            
        tensor_score = tensor_score.to(device)

        # print("tensor_score")
        # print(tensor_score[0])

        current_score = torch.max(tensor_score, dim=1).values

        # for i in range(100):
        #     print("current_score : " + str(i))
        #     print(current_score[i])

        # print(coucou)
        bestScore = torch.where(current_score > bestScore, current_score, bestScore)


        # print("tensor_solution.size()")
        # print(tensor_solution.size())

        strategy.updateDistribution(tensor_solution , tensor_score)


        if(verbose):
            pbar.set_postfix(bestScore=torch.mean(bestScore).item(),
                             current_score=torch.mean(current_score).item())

        if (name_file is not None):
            if (((epoch + 1) * size_pop) % 100 == 0):
                bestScore_np = bestScore.cpu().numpy()
                mean = np.mean(bestScore_np)
                median = np.percentile(bestScore_np, 50)
                std = np.std(bestScore_np)
                _2per = np.percentile(bestScore_np, 2)
                _5per = np.percentile(bestScore_np, 5)
                _10per = np.percentile(bestScore_np, 10)
                _25per = np.percentile(bestScore_np, 25)
                _75per = np.percentile(bestScore_np, 75)
                _90per = np.percentile(bestScore_np, 90)
                _95per = np.percentile(bestScore_np, 95)
                _98per = np.percentile(bestScore_np, 98)

                f_results = open(name_file, "a")
                f_results.write(
                    str((epoch + 1) * size_pop) + "," + str(mean) + "," + str(median) + "," + str(std) + "," + str(
                        _2per) + "," + str(_5per) + "," + str(_10per) + "," + str(_25per) + "," + str(
                        median) + "," + str(_75per) + "," + str(_90per) + "," + str(_95per) + "," + str(_98per) + "\n")
                f_results.close()



    return bestScore.cpu().numpy()
