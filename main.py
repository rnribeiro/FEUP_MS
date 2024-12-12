import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time


def main():
    # Configuração do ambiente
    # register new myenv.py environment
    gym.envs.register(
        id='myenv',
        entry_point='myenv:MyEnv',
    )

    env = gym.make('myenv', render_mode='rgb_array')
    
    # Criação do modelo
    ppo_model = PPO('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              batch_size=128,
              gamma=0.8,
              n_steps=2048,  # Number of steps to run for each environment per update
              ent_coef=0.01,  # Entropy coefficient
              vf_coef=0.5,  # Value function coefficient
              max_grad_norm=0.5,  # Maximum gradient norm
              gae_lambda=0.95,  # GAE lambda
              clip_range=0.2,  # Clipping range
              verbose=1,
              tensorboard_log="logs/",
              device='cuda')

    dqn_model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqn/",
            device='cuda')


    model.learn(int(1e3))

    # Salvar o modelo treinado
    model.save("ppo_highway_test")

    """model = PPO.load("ppo_highway_test")"""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Average reward: {mean_reward} ± {std_reward}")
    # Demonstração
    obs, _ = env.reset()  
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info, truncation = env.step(action)  
        env.render()
        time.sleep(1/20)  # Add a delay to achieve 30 fps
        if done:
            obs, _ = env.reset() 
    
    env.close()

if __name__ == "__main__":
    main()
