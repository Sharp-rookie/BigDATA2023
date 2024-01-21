"""
智能体评估函数
"""
import torch
from ding.utils import set_pkg_seed
from mario_dqn_config import mario_dqn_config, mario_dqn_create_config
from model import DQN
from policy import DQNPolicy
from ding.config import compile_config
from ding.envs import DingEnvWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv, RecordCAM, SparseRewardWrapper, CoinRewardWrapper, StickyActionWrapper, JumpRewardWrapper, ScoreRewardWrapper, WallHitPenaltyWrapper
import warnings; warnings.filterwarnings("ignore")
import numpy as np


action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]

def wrapped_mario_env(model, cam_video_path, version=0, action=2, obs=1, my_wrapper=0, size=84, skip=4, stage=1):
    
    joypadspace = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-"+str(stage)+"-v"+str(version)), action_dict[int(action)])
    cfg={
        'env_wrapper': [
            lambda env: MaxAndSkipWrapper(env, skip=skip),
            lambda env: WarpFrameWrapper(env, size=size),
            lambda env: ScaledFloatFrameWrapper(env),
            lambda env: FrameStackWrapper(env, n_frames=obs),
            lambda env: FinalEvalRewardEnv(env),
            lambda env: RecordCAM(env, cam_model=model, video_folder=cam_video_path)
        ]
    }
    
    # 添加自定义wrapper
    if my_wrapper == 1:
        # 金币奖励wrapper
        cfg['env_wrapper'].append(lambda env: CoinRewardWrapper(env))
    elif my_wrapper == 2:
        # 稀疏奖励wrapper
        cfg['env_wrapper'].append(lambda env: SparseRewardWrapper(env))
    elif my_wrapper == 3:
        # 粘性动作wrapper
        cfg['env_wrapper'].append(lambda env: StickyActionWrapper(env))
    elif my_wrapper == 4:
        # 跳跃奖励wrapper
        cfg['env_wrapper'].append(lambda env: JumpRewardWrapper(env))
    elif my_wrapper == 5:
        # 行为奖励wrapper
        cfg['env_wrapper'].append(lambda env: ScoreRewardWrapper(env))
    elif my_wrapper == 6:
        # 撞墙惩罚wrapper
        cfg['env_wrapper'].append(lambda env: WallHitPenaltyWrapper(env))
    
    return DingEnvWrapper(joypadspace, cfg)


def evaluate(args, state_dict, video_dir_path, eval_times):
    # 加载配置
    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)
    # 实例化DQN模型
    model = DQN(**cfg.policy.model)
    # 加载模型权重文件
    model.load_state_dict(state_dict['model'])
    # 实例化DQN策略
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    
    avg_reward_list = []
    for n, seed in enumerate(range(1, eval_times+1)):
        # 生成环境
        env = wrapped_mario_env(model, video_dir_path, args.version, args.action, args.obs, my_wrapper=args.my_wrapper, size=args.img_size, skip=args.skip, stage=args.stage)
        # 保存录像
        if n == 0:
            env.enable_save_replay(video_dir_path)
        # 设置seed
        env.seed(seed)
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
        eval_reward_list = []
        # 环境重置，返回初始观测
        obs = env.reset()
        eval_reward = 0
        while True:
            # 策略根据观测返回所有动作的Q值以及Q值最大的动作
            Q = policy.forward({0: obs})
            # 获取动作
            action = Q[0]['action'].item()
            # 将动作传入环境，环境返回下一帧信息
            obs, reward, done, info = env.step(action)
            eval_reward += reward
            if done or info['time'] < 250:
                print(info)
                eval_reward_list.append(eval_reward)
                break
        avg_reward_list.append(sum(eval_reward_list) / len(eval_reward_list))
        print('During {}th evaluation, the total reward your mario got is {}'.format(n, eval_reward))
        
        try:
            del env
        except Exception:
            pass
        
        np.savez(video_dir_path + f'/eval_metrics_{n}.npz', eval_reward=eval_reward, info=info)
    
    print('Eval is over! The performance of your RL policy is {}(±{})'.format(np.mean(avg_reward_list), np.std(avg_reward_list)))
    print("Your mario video is saved in {}".format(video_dir_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-sd", type=int, default=0)
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,12])
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])
    parser.add_argument("--my_wrapper", "-m", type=int, default=0, choices=[0,1,2,3,4,5,6])
    parser.add_argument("--img_size", "-i", type=int, default=84, choices=[84,42])
    parser.add_argument("--skip", "-sk", type=int, default=4, choices=[4,8])
    parser.add_argument("--stage", "-st", type=int, default=1, choices=[1,2,3])
    parser.add_argument("--duel", "-d", type=int, default=0, choices=[0,1])
    parser.add_argument("--noise", "-n", type=int, default=0, choices=[0,1])
    args = parser.parse_args()
    
    mario_dqn_config.policy.model.obs_shape=[args.obs, args.img_size, args.img_size]
    mario_dqn_config.policy.model.action_shape=args.action
    mario_dqn_config.policy.model.dueling=bool(args.duel)
    mario_dqn_config.policy.model.noise=bool(args.noise)
    mario_dqn_config.exp_name = f'exp/duel{args.duel}_noisy{args.noise}/v{args.version}_a{args.action}_o{args.obs}_i{args.img_size}_s{args.skip}_m{args.my_wrapper}_stage{args.stage}_seed{args.seed}'
    ckpt_path = mario_dqn_config.exp_name + '/ckpt/ckpt_best.pth.tar'
    video_dir_path = mario_dqn_config.exp_name + '/eval_videos'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    evaluate(args, state_dict=state_dict, video_dir_path=video_dir_path, eval_times=5)
