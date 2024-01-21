import os
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
plt.style.use (['science'])


def smooth(data, weight=0.95):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def filename2name(filename):
    name = filename.split('_st')[0]
    if 'v1' in name:
        name = 'no sky'
    elif 'a2' in name:
        name = '2 actions'
    elif 'a12' in name:
        name = '12 actions'
    elif 'o4' in name:
        name = '4 observations'
    elif 'i42' in name:
        name = 'downsample'
    elif 'm1' in name:
        name = 'coin reward'
    elif 'm2' in name:
        name = 'sparse reward'
    elif 'm3' in name:
        name = 'sticky action'
    elif 's8' in name:
        name = '8 skip frames'
    else:
        name = 'default'
    return name


def step1(stage=1):
    
    exp_dir = "exp/duel0_noisy0/"
    
    exp_names = [
        f'v0_a7_o1_i42_s4_m0_stage{stage}_seed612',
        f'v0_a12_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a2_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m0_stage{stage}_seed612',
        f'v1_a7_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o4_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s8_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m1_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m2_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m3_stage{stage}_seed612',
    ]

    plt.figure(figsize=(6, 6))
    episode_n_list = []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name:
            continue

        log_file_path = exp_dir + exp_name + "/log/evaluator/evaluator_logger.txt"

        reward_means = []
        with open(log_file_path, 'r') as file:
            
            count, target = 0, -1
            for line in file:
                
                if "reward_mean" in line:
                    target = count + 2
                    
                if count == target:
                    reward_mean = float(line.split("|")[2].strip())
                    reward_means.append(reward_mean)
                
                count += 1

        logs = range(1, len(reward_means) + 1)
        
        # 统计收敛所需episode
        if 'm2' not in exp_name:
            episode = 0
            for reward_mean in reward_means:
                if reward_mean > (3000 if stage == 1 else 4500):
                    break
                episode += 1
            episode_n_list.append(episode)

        plt.plot(logs, smooth(reward_means), label=filename2name(exp_name), linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.8)
        plt.scatter(logs[-1], smooth(reward_means)[-1], s=50, c='black')
        # plt.text(logs[-1]-75, smooth(reward_means)[-1]+10, filename2name(exp_name), fontsize=16, fontweight='bold')

        plt.title('Reward per Iter (smooth weight = 0.95)', fontsize=20)
        plt.xlabel('Iter', fontsize=20)
        plt.ylabel('Value', fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"reward_mean_iter_st{stage}.pdf", dpi=300)


    plt.figure(figsize=(5, 5))
    reward_n_list, coins_n_list, score_n_list, time_n_list, flag_n_list = [], [], [], [], []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue

        reward_n, coins_n, score_n, time_n, flag_n = [], [], [], [], []
        for n in range(5):
            log_file_path = exp_dir + exp_name + f"/eval_videos/eval_metrics_{n}.npz"
            metrics = np.load(log_file_path, allow_pickle=True)
            reward_n.append(metrics['eval_reward'])
            info = metrics['info'].item()
            coins_n.append(info['coins'])
            score_n.append(info['score'])
            time_n.append(info['time'] if info['flag_get'] else 0)
            flag_n.append(int(info['flag_get']))
        
        reward_n_list.append(np.mean(reward_n))
        coins_n_list.append(np.mean(coins_n))
        score_n_list.append(np.mean(score_n))
        time_n_list.append(np.mean(time_n))
        flag_n_list.append(np.mean(flag_n))

    i = 0
    convergence_speeds = [np.max(episode_n_list) - episode for episode in episode_n_list]
    for exp_name in exp_names:
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue
        
        reward = (reward_n_list[i] - np.mean(reward_n_list)) / np.std(reward_n_list)
        coins = (coins_n_list[i] - np.mean(coins_n_list)) / np.std(coins_n_list)
        convergence_speed = (convergence_speeds[i] - np.mean(convergence_speeds)) / (np.std(convergence_speeds) + 1e-8)
        score = (score_n_list[i] - np.mean(score_n_list)) / np.std(score_n_list)
        time = (time_n_list[i] - np.mean(time_n_list)) / np.std(time_n_list)
        flag = (flag_n_list[i] - np.mean(flag_n_list)) / (np.std(flag_n_list) + 1e-8)
        
        # 雷达图
        properties = np.array(['reward', 'coins', 'convergence speed', 'score', 'speed', 'flag'])
        values = np.array([reward, coins, convergence_speed, score, time, flag])
        angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False)
        plt.polar(np.concatenate((angles,[angles[0]])), np.concatenate((values,[values[0]])), label=filename2name(exp_name), linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.8)
        plt.xticks(angles, properties, fontsize=18)
        plt.yticks([], [])
        
        i += 1
    
    # legend显示在图像外
    # plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=16)
    # plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"radar_st{stage}.pdf", dpi=300)


def step2(stage=1):
    
    exp_dir = "exp/duel0/"
    
    exp_names = [
        f'v0_a7_o1_i42_s4_m0_stage{stage}_seed612',
        f'v0_a12_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a2_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m0_stage{stage}_seed612',
        f'v1_a7_o1_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o4_i84_s4_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s8_m0_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m1_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m2_stage{stage}_seed612',
        f'v0_a7_o1_i84_s4_m3_stage{stage}_seed612',
    ]
    
    if stage == 1:
        best_exp_path = 'exp/duel0/v0_a2_o4_i84_s4_m0_stage1_seed612'
        best_exp_path2 = 'exp/duel0_noisy0/v1_a2_o4_i84_s4_m0_stage1_seed612'

    plt.figure(figsize=(6, 6))
    episode_n_list = []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name:
            continue

        log_file_path = exp_dir + exp_name + "/log/evaluator/evaluator_logger.txt"

        reward_means = []
        with open(log_file_path, 'r') as file:
            
            count, target = 0, -1
            for line in file:
                
                if "reward_mean" in line:
                    target = count + 2
                    
                if count == target:
                    reward_mean = float(line.split("|")[2].strip())
                    reward_means.append(reward_mean)
                
                count += 1

        logs = range(1, len(reward_means) + 1)
        
        # 统计收敛所需episode
        if 'm2' not in exp_name:
            episode = 0
            for reward_mean in reward_means:
                if reward_mean > (3000 if stage == 1 else 4500):
                    break
                episode += 1
            episode_n_list.append(episode)

        plt.plot(logs, smooth(reward_means), label=filename2name(exp_name), linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.3)
        plt.scatter(logs[-1], smooth(reward_means)[-1], s=50, c='black')
        # plt.text(logs[-1]-75, smooth(reward_means)[-1]+10, filename2name(exp_name), fontsize=16, fontweight='bold')

        plt.title('Reward per Iter (smooth weight = 0.95)', fontsize=20)
        plt.xlabel('Iter', fontsize=20)
        plt.ylabel('Value', fontsize=20)
    
    log_file_path = best_exp_path + "/log/evaluator/evaluator_logger.txt"
    reward_means = []
    with open(log_file_path, 'r') as file:
        count, target = 0, -1
        for line in file:
            if "reward_mean" in line:
                target = count + 2
            if count == target:
                reward_mean = float(line.split("|")[2].strip())
                reward_means.append(reward_mean)
            count += 1
    logs = range(1, len(reward_means) + 1)
    # 统计收敛所需episode
    if 'm2' not in exp_name:
        episode = 0
        for reward_mean in reward_means:
            if reward_mean > (3000 if stage == 1 else 4500):
                break
            episode += 1
        episode_n_list.append(episode)
    plt.plot(logs, smooth(reward_means), label='Best', linewidth=2.0, alpha=1)
    plt.scatter(logs[-1], smooth(reward_means)[-1], s=50, c='black')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"reward_mean_iter_st{stage}_best.pdf", dpi=300)


    plt.figure(figsize=(5, 5))
    reward_n_list, coins_n_list, score_n_list, time_n_list, flag_n_list = [], [], [], [], []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue

        reward_n, coins_n, score_n, time_n, flag_n = [], [], [], [], []
        for n in range(5):
            log_file_path = exp_dir + exp_name + f"/eval_videos/eval_metrics_{n}.npz"
            metrics = np.load(log_file_path, allow_pickle=True)
            reward_n.append(metrics['eval_reward'])
            info = metrics['info'].item()
            coins_n.append(info['coins'])
            score_n.append(info['score'])
            time_n.append(info['time'] if info['flag_get'] else 0)
            flag_n.append(int(info['flag_get']))
        
        reward_n_list.append(np.mean(reward_n))
        coins_n_list.append(np.mean(coins_n))
        score_n_list.append(np.mean(score_n))
        time_n_list.append(np.mean(time_n))
        flag_n_list.append(np.mean(flag_n))

    i = 0
    convergence_speeds = [np.max(episode_n_list) - episode for episode in episode_n_list]
    for exp_name in exp_names:
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue
        
        reward = (reward_n_list[i] - np.mean(reward_n_list)) / np.std(reward_n_list)
        coins = (coins_n_list[i] - np.mean(coins_n_list)) / np.std(coins_n_list)
        convergence_speed = (convergence_speeds[i] - np.mean(convergence_speeds)) / (np.std(convergence_speeds) + 1e-8)
        score = (score_n_list[i] - np.mean(score_n_list)) / np.std(score_n_list)
        time = (time_n_list[i] - np.mean(time_n_list)) / np.std(time_n_list)
        flag = (flag_n_list[i] - np.mean(flag_n_list)) / (np.std(flag_n_list) + 1e-8)
        
        # 雷达图
        properties = np.array(['reward', 'coins', 'convergence speed', 'score', 'speed', 'flag'])
        values = np.array([reward, coins, convergence_speed, score, time, flag])
        angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False)
        plt.polar(np.concatenate((angles,[angles[0]])), np.concatenate((values,[values[0]])), label=filename2name(exp_name), linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.3)
        plt.xticks(angles, properties, fontsize=18)
        plt.yticks([], [])
        
        i += 1
    
    for i, best_path in enumerate([best_exp_path, best_exp_path2]):
        reward_n, coins_n, score_n, time_n, flag_n = [], [], [], [], []
        for n in range(5):
            log_file_path = best_path + f"/eval_videos/eval_metrics_{n}.npz"
            metrics = np.load(log_file_path, allow_pickle=True)
            reward_n.append(metrics['eval_reward'])
            info = metrics['info'].item()
            coins_n.append(info['coins'])
            score_n.append(info['score'])
            time_n.append(info['time'] if info['flag_get'] else 0)
            flag_n.append(int(info['flag_get']))
        best_reward = (np.mean(reward_n) - np.mean(reward_n_list)) / np.std(reward_n_list)
        best_coins = (np.mean(coins_n) - np.mean(coins_n_list)) / np.std(coins_n_list)
        best_convergence_speed = (np.max(episode_n_list) - np.mean(convergence_speeds)) / (np.std(convergence_speeds) + 1e-8)
        best_score = (np.mean(score_n) - np.mean(score_n_list)) / np.std(score_n_list)
        best_time = (np.mean(time_n) - np.mean(time_n_list)) / np.std(time_n_list)
        best_flag = (np.mean(flag_n) - np.mean(flag_n_list)) / (np.std(flag_n_list) + 1e-8)
        values = np.array([best_reward, best_coins, best_convergence_speed, best_score, best_time, best_flag])
        angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False)
        plt.polar(np.concatenate((angles,[angles[0]])), np.concatenate((values,[values[0]])), linewidth=2.0, alpha=1, c=['red', 'blue'][i])
    
    # legend显示在图像外
    # plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=16)
    # plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"radar_st{stage}_best.pdf", dpi=300)


def step3(stage=1):
    """测试Algorithm Advances"""
    
    baseline_path = f'exp/duel0_noisy0/v0_a7_o1_i84_s4_m0_stage{stage}_seed612'
    dueling_path = f'exp/duel1_noisy0/v0_a7_o1_i84_s4_m0_stage{stage}_seed612'
    noisy_path = f'exp/duel0_noisy1/v0_a7_o1_i84_s4_m0_stage{stage}_seed612'
    rainbow_path = f'exp/duel1_noisy1/v0_a7_o1_i84_s4_m0_stage{stage}_seed612'
    
    exp_names = [baseline_path, dueling_path, noisy_path, rainbow_path]

    episode_n_list = []
    for i, exp_name in enumerate(exp_names):

        log_file_path = exp_name + "/log/evaluator/evaluator_logger.txt"

        reward_means = []
        with open(log_file_path, 'r') as file:
            
            count, target = 0, -1
            for line in file:
                
                if "reward_mean" in line:
                    target = count + 2
                    
                if count == target:
                    reward_mean = float(line.split("|")[2].strip())
                    reward_means.append(reward_mean)
                
                count += 1
        
        # 统计收敛所需episode
        if 'm2' not in exp_name:
            episode = 0
            for reward_mean in reward_means:
                if reward_mean > (3000 if stage == 1 else 4500):
                    break
                episode += 1
            episode_n_list.append(episode)


    reward_n_list, coins_n_list, score_n_list, time_n_list, flag_n_list = [], [], [], [], []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue

        reward_n, coins_n, score_n, time_n, flag_n = [], [], [], [], []
        for n in range(5):
            log_file_path = exp_name + f"/eval_videos/eval_metrics_{n}.npz"
            metrics = np.load(log_file_path, allow_pickle=True)
            reward_n.append(metrics['eval_reward'])
            info = metrics['info'].item()
            coins_n.append(info['coins'])
            score_n.append(info['score'])
            time_n.append(info['time'] if info['flag_get'] else 0)
            flag_n.append(int(info['flag_get']))
        
        reward_n_list.append(np.mean(reward_n))
        coins_n_list.append(np.mean(coins_n))
        score_n_list.append(np.mean(score_n))
        time_n_list.append(np.mean(time_n))
        flag_n_list.append(np.mean(flag_n))

    for i, algo in enumerate(['Baseline', 'Dueling DQN', 'Noisy DQN', 'Rainbow DQN']):
        print(f'{algo} & 训练轮数 {episode_n_list[i]} & 平均奖励 {reward_n_list[i]:.2f} & 金币 {coins_n_list[i]:.2f} & 得分 {score_n_list[i]:.2f} & 耗时 {400-time_n_list[i]:.2f} & 通过率 {flag_n_list[i]:.2f} \\\\')


def step4(stage=1):
    """测试Reward Shaping"""
    
    baseline_path = f'exp/duel0_noisy0/v0_a7_o1_i84_s4_m0_stage{stage}_seed612'
    jump_path = f'exp/duel0_noisy0/v0_a7_o1_i84_s4_m4_stage{stage}_seed612'
    wall_path = f'exp/duel0_noisy0/v0_a7_o1_i84_s4_m6_stage{stage}_seed612'
    score_path = f'exp/duel0_noisy0/v0_a7_o1_i84_s4_m5_stage{stage}_seed612'
    labels = ['Default', 'Jump', 'Wall Hit', 'Score']
    
    exp_names = [baseline_path, jump_path, wall_path, score_path]

    plt.figure(figsize=(6, 6))
    episode_n_list = []
    for i, exp_name in enumerate(exp_names):

        log_file_path = exp_name + "/log/evaluator/evaluator_logger.txt"

        reward_means = []
        with open(log_file_path, 'r') as file:
            
            count, target = 0, -1
            for line in file:
                
                if "reward_mean" in line:
                    target = count + 2
                    
                if count == target:
                    reward_mean = float(line.split("|")[2].strip())
                    reward_means.append(reward_mean)
                
                count += 1
        
        logs = range(1, len(reward_means) + 1)
        
        # 统计收敛所需episode
        if 'm2' not in exp_name:
            episode = 0
            for reward_mean in reward_means:
                if reward_mean > (3000 if stage == 1 else 4500):
                    break
                episode += 1
            episode_n_list.append(episode)
        
        plt.plot(logs, smooth(reward_means), label=labels[i], linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.8)
        plt.scatter(logs[-1], smooth(reward_means)[-1], s=50, c='black')
        # plt.text(logs[-1]-75, smooth(reward_means)[-1]+10, filename2name(exp_name), fontsize=16, fontweight='bold')

        plt.title('Reward per Iter (smooth weight = 0.95)', fontsize=20)
        plt.xlabel('Iter', fontsize=20)
        plt.ylabel('Value', fontsize=20)
        
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"reward_mean_iter_st{stage}_rs.pdf", dpi=300)
    
    plt.figure(figsize=(5, 5))
    reward_n_list, coins_n_list, score_n_list, time_n_list, flag_n_list = [], [], [], [], []
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue

        reward_n, coins_n, score_n, time_n, flag_n = [], [], [], [], []
        for n in range(5):
            log_file_path = exp_name + f"/eval_videos/eval_metrics_{n}.npz"
            metrics = np.load(log_file_path, allow_pickle=True)
            reward_n.append(metrics['eval_reward'])
            info = metrics['info'].item()
            coins_n.append(info['coins'])
            score_n.append(info['score'])
            time_n.append(info['time'] if info['flag_get'] else 0)
            flag_n.append(int(info['flag_get']))
        
        reward_n_list.append(np.mean(reward_n))
        coins_n_list.append(np.mean(coins_n))
        score_n_list.append(np.mean(score_n))
        time_n_list.append(np.mean(time_n))
        flag_n_list.append(np.mean(flag_n))

    i = 0
    best_value = []
    convergence_speeds = [np.max(episode_n_list) - episode for episode in episode_n_list]
    for i, exp_name in enumerate(exp_names):
        
        if f'_stage{stage}_' not in exp_name or 'm2' in exp_name:
            continue
        
        reward = (reward_n_list[i] - np.mean(reward_n_list)) / np.std(reward_n_list)
        coins = (coins_n_list[i] - np.mean(coins_n_list)) / np.std(coins_n_list)
        convergence_speed = (convergence_speeds[i] - np.mean(convergence_speeds)) / (np.std(convergence_speeds) + 1e-8)
        score = (score_n_list[i] - np.mean(score_n_list)) / np.std(score_n_list)
        time = (time_n_list[i] - np.mean(time_n_list)) / np.std(time_n_list)
        flag = (flag_n_list[i] - np.mean(flag_n_list)) / (np.std(flag_n_list) + 1e-8)
        
        if i == 0:
            best_value = [reward, coins, convergence_speed, score, time, flag]
        else:
            for item in [reward, coins, convergence_speed, score, time, flag]:
                if item > best_value[i]:
                    best_value[i] = item

        # 雷达图
        properties = np.array(['reward', 'coins', 'convergence speed', 'score', 'speed', 'flag'])
        values = np.array([reward, coins, convergence_speed, score, time, flag])
        angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False)
        plt.polar(np.concatenate((angles,[angles[0]])), np.concatenate((values,[values[0]])), label=labels[i], linewidth=1.6, linestyle='-.' if i % 2 == 0 else '-', alpha=0.5)
        plt.xticks(angles, properties, fontsize=18)
        plt.yticks([], [])
        
        i += 1
        
    # 雷达图
    properties = np.array(['reward', 'coins', 'convergence speed', 'score', 'speed', 'flag'])
    values = np.array(best_value)
    angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False)
    plt.polar(np.concatenate((angles,[angles[0]])), np.concatenate((values,[values[0]])), linewidth=2.0, alpha=1, c='red')
    
    # legend显示在图像外
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=16)
    # plt.legend(frameon=False, fontsize=16)
    plt.savefig(f"radar_st{stage}_rs.pdf", dpi=300)


step1()
# step2()
# step3(stage=1)
# step4()