
import random
import time
import os

from setting import parse_opt

sets = parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = sets.gpu

import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from train.local_train import Agent
from aggregation.agg import Aggregation, Aggregation_aux
from medpy.metric.binary import dc, hd95
from model.unet_model import UNet2








def seed_torch(seed=666):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
seed_torch()

sets = parse_opt()

def train():
    
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
   
    
    global_model = nn.DataParallel(UNet2(1, sets.num_classes)).cuda()
    global_model = global_model.to(device)
  
    
   
    agents, agent_data_size = [], {}
    for agent_id in range(sets.num_agents):
        agent = Agent(agent_id)
        agent_data_size[agent_id] = agent.n_data
        agents.append(agent)
    n_model_params = len(parameters_to_vector(global_model.parameters()))
   
    aggregator = Aggregation(agent_data_size, n_model_params)
    aggregator_aux = Aggregation_aux(agent_data_size,n_model_params)
    best_dice = 0.0
    global_model = global_model.float()
    aux_model = copy.deepcopy(global_model)
    aux_model.module = aux_model.module.float()

   
    for rnd in range(1, sets.rounds + 1):
        lam = 1 / (np.exp(40 - rnd) + 1)
        epoch_start = time.time()
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}
        binary_outputs_dict = {}  
        labeled_updates={}
        consistency_stats_dict = {}
        
        
        
        all_agents = list(range(sets.num_agents))
        labeled_agents = all_agents[:4]
        unlabeled_agents = all_agents[4:6]
        
            
        
        for agent_id in tqdm(labeled_agents):
            update1, binary_output = agents[agent_id].train(global_model, rnd,aux_model,labeled=True)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
            agent_updates_dict[agent_id] = update1
            binary_outputs_dict[agent_id] = binary_output
            labeled_updates[agent_id]  = update1
           

      
        for agent_id in tqdm(unlabeled_agents):
            update1, binary_output = agents[agent_id].train(global_model, rnd,aux_model,labeled=False)
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
            agent_updates_dict[agent_id] = update1
            binary_outputs_dict[agent_id] = binary_output
    

        
        labeled_updates = [agent_updates_dict[aid].float() for aid in labeled_agents] 
        num_labeled = len(labeled_agents)
        
        if num_labeled > 0:
            aggregator_aux.aggregate_update(aux_model, labeled_updates)
        
       
        aggregator.aggregate_update(global_model, agent_updates_dict, binary_outputs_dict)
        

  
    


        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        dice1 = 0.0
        dice2 = 0.0
        dice3 = 0.0
        jaccard1 = 0.0
        jaccard2 = 0.0
        jaccard3 = 0.0
        sensitive1 = 0.0
        sensitive2 = 0.0
        sensitive3 = 0.0
      


        with torch.no_grad():
            global_model.eval()

            total_loss = 0.0
            total_dice = 0.0
            total_jaccard = 0.0
            total_sensitive = 0.0

            for vid in range(6):
                valid_data = dataset.dataload.get_train_data_loader(1, f"valid_data/train_data_{vid}")

                loss1 = loss2 = loss3 = 0.0
                dice1 = dice2 = dice3 = 0.0
                jaccard1 = jaccard2 = jaccard3 = 0.0
                sensitive1 = sensitive2 = sensitive3 = 0.0

                sample_count = 0  

                for j, (inputs, labels) in enumerate(valid_data):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    sample_count += inputs.size(0)
                    out_1, out_2, feat = global_model(inputs)
                    _, pre = torch.max(out_2, 1)
                    res1 = torch.where(pre == 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    res2 = torch.where(pre == 1, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    res3 = torch.where(pre == 2, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    res4 = torch.where(pre == 3, torch.tensor(1).to(device), torch.tensor(0).to(device))
                    res = torch.cat((res1.unsqueeze(1), res2.unsqueeze(1), res3.unsqueeze(1), res4.unsqueeze(1)), 1)
                    dice1 += Dice_Coeff(res, labels, 1) * inputs.size(0)
                    dice2 += Dice_Coeff(res, labels, 2) * inputs.size(0)
                    dice3 += Dice_Coeff(res, labels, 3) * inputs.size(0)
                avg_valid_dice = (dice1 + dice2 + dice3) / (3 * sample_count)
                print(f"ValidSet{vid}:  Dice {avg_valid_dice:.4f} ")
                total_dice += avg_valid_dice
            mean_dice = total_dice / 6
            if best_dice < mean_dice:
                best_dice = mean_dice
                save_path = f"fedrnp_{sets.gpu}_.pth"
                torch.save(global_model.state_dict(), save_path)
            epoch_end = time.time()
            print("Best Dice:", best_dice.item())









def DiceCoeff(pred, target):
    smooth = 0.0001
    ifflat = pred.contiguous().view(-1)
    tfflat = target.contiguous().view(-1)
    intersection = (ifflat * tfflat).sum()
    score = ((2 * intersection + smooth) / (ifflat.sum() + tfflat.sum() + smooth))
    
    return score

def Dice_Coeff(input, target, n):
    """Dice coeff for batches"""
    dice = 0
    for i in range(input.shape[0]):
        dice += DiceCoeff(input[i][n], target[i][n])
    return dice / input.shape[0]




def calculate_average_hd95_batch(pred_batch, gt_batch, num_classes=4, penalty_value=400.0):
    batch_size = pred_batch.shape[0]
    class_hd95 = {c: [] for c in range(1, num_classes)}
    for i in range(batch_size):
        pred = pred_batch[i]
        gt = gt_batch[i]
        for c in range(1, num_classes):
            pred_c = (pred == c).astype(np.bool_)
            gt_c = (gt == c).astype(np.bool_)
            pred_sum = np.sum(pred_c)
            gt_sum = np.sum(gt_c)
            if pred_sum == 0 and gt_sum == 0:
                class_hd95[c].append(0.0)
                continue
            if pred_sum == 0 or gt_sum == 0:
                class_hd95[c].append(penalty_value)
                continue
            try:
                value = hd95(pred_c, gt_c)
                class_hd95[c].append(value)
            except Exception as e:
                print(f"⚠️ Error on sample {i}, class {c}: {e}")
                class_hd95[c].append(penalty_value)
    results = {}
    for c in range(1, num_classes):
        values = class_hd95[c]
        if len(values) == 0:
            results[c] = None
        else:
            results[c] = float(np.mean(values))

    return results

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = nn.DataParallel(UNet2(1, sets.num_classes)).cuda()
    load_path = f"fedrnp_{sets.gpu}_.pth"
    global_model.load_state_dict(torch.load(load_path))
    global_model = global_model.to(device)

    with torch.no_grad():
        global_model.eval()
        test_results = []
        for i in range(6):
            print(f"\n========== 测试集{i} ==========")
            test_data = dataset.dataload.get_train_data_loader(1, "test_data/train_data_" + str(i))
            dice1 = dice2 = dice3 = 0.0
            total_samples = 0
            num_classes = 4  
            all_hd = {c: 0.0 for c in range(1, num_classes)}   
            count_hd = {c: 0  for c in range(1, num_classes)}
            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                out_1, out_2, feat = global_model(inputs)
                _, pre = torch.max(out_2, 1)
                _, label = torch.max(labels, 1)
                hd = calculate_average_hd95_batch(np.array(pre.cpu()), np.array(label.cpu()))
                res1 = torch.where(pre == 0, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res2 = torch.where(pre == 1, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res3 = torch.where(pre == 2, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res4 = torch.where(pre == 3, torch.tensor(1).to(device), torch.tensor(0).to(device))
                res = torch.cat((res1.unsqueeze(1), res2.unsqueeze(1), res3.unsqueeze(1), res4.unsqueeze(1)), 1)
                batch_size = inputs.size(0)
                total_samples += batch_size
                dice1 += Dice_Coeff(res, labels, 1) * batch_size
                dice2 += Dice_Coeff(res, labels, 2) * batch_size
                dice3 += Dice_Coeff(res, labels, 3) * batch_size
                for c, v in hd.items():
                    if v is not None:
                        all_hd[c] += v * batch_size
                        count_hd[c] += batch_size
            avg_valid_dice1 = dice1 / total_samples
            avg_valid_dice2 = dice2 / total_samples
            avg_valid_dice3 = dice3 / total_samples
            avg_valid_dice = (avg_valid_dice1 + avg_valid_dice2 + avg_valid_dice3) / 3
            avg_hd = {
                c: (all_hd[c] / count_hd[c]) if count_hd[c] > 0 else None
                for c in all_hd
            }

            test_results.append({
                "test_set": i,
                "dice": {1: avg_valid_dice1, 2: avg_valid_dice2, 3: avg_valid_dice3},
                "hd": avg_hd,
            })           
            print(f"\n=== TestSet{i} 结果 ===")
            print(" Dice: {:.4f}".format(
                 avg_valid_dice))
            print("HD95:")
            for c in avg_hd:
                if avg_hd[c] is None:
                    print(f"  - 类{c}: None")
                else:
                    print(f"  - 类{c}: {avg_hd[c]:.4f}")       
        classes = [1, 2, 3] 
        dice_sum = {c: 0.0 for c in classes}
        hd_sum = {c: 0.0 for c in classes}
        count = 0   
        for r in test_results:
            for c in classes:
                dice_sum[c] += r["dice"][c]
                hd_sum[c] += r["hd"][c]
            count += 1    
        avg_dice = {c: dice_sum[c] / count for c in classes}
        avg_hd = {c: hd_sum[c] / count for c in classes}       
        mean_dice = sum(avg_dice.values()) / len(classes)
        mean_hd = sum(avg_hd.values()) / len(classes)
        print("\n=== 各类指标平均 ===")
        for c in classes:
            print(f"类{c} ->  Dice: {avg_dice[c]:.4f}, HD95: {avg_hd[c]:.4f}")
        print("\n=== 所有前景类整体平均 ===")
        print(f"Dice: {mean_dice:.4f}, HD95: {mean_hd:.4f}")
        





    
if __name__ == '__main__':
    train()
    test()
    
