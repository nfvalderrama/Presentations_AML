import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym

# parser for all parameters and hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str)
parser.add_argument("--env_name", default="MountainCar-v0")

parser.add_argument('--q_network_iteration', default=100, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=10e-3, type=float) # same for critic and actor
parser.add_argument('--gamma', default=0.9, type=int) # discount factor (is high cause dynamics are super fast)
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size (should change based on your resources)
parser.add_argument('--max_length_of_trajectory', default=10000, type=int) # length of game

parser.add_argument('--batch_size', default=32, type=int) # mini batch size
parser.add_argument('--epsilon', default=0.9, type=int)

# optional parameters
parser.add_argument('--render', default=False, type=bool) # show UI (best your for test)
parser.add_argument('--show_performance', default=True, type=bool) # show performance
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load pre-trained model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work

parser.add_argument('--max_episode', default=5, type=int) # num of episodes for train

args = parser.parse_args()


######################################
#environment


device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = 'DQN'

env = gym.make(args.env_name).unwrapped

state_dim = env.observation_space.shape[0] #2
action_dim = env.action_space.n

min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './Trained_Models/' + script_name + '_' + args.env_name +'/'


#network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, action_dim)
        self.fc2.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

#dqn model
class Dqn():
    #init of the networks, memory space, some counters and both optimizer and loss
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = np.zeros((args.max_length_of_trajectory, state_dim *2 +2))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), args.learning_rate)
        self.loss = nn.MSELoss()

        self.fig, self.ax = plt.subplots()

    #store sequences into memory
    def store_trans(self, state, action, reward, next_state):
        #if self.memory_counter % 500 ==0:
        #    print(f"The experience pool collects {self.memory_counter} time experience")
        index = self.memory_counter % args.max_length_of_trajectory
        trans = np.hstack((state, [action], [reward], next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    #choose action based on a state and epsilon-greedy function
    def choose_action(self, state):
        #notation that the function return the action's index nor the real action
        state = torch.unsqueeze(torch.FloatTensor(state) ,0)
        if np.random.randn() <= args.epsilon:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy() #get action whose q is max
            action = action[0] #get the action index
        else:
            action = np.random.randint(0,action_dim)
        return action

    #plot total reward vs episode
    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    #q update based on discounted return
    def learn(self):
        #learn 100 times then the target network update
        if self.learn_counter % args.q_network_iteration ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1
        
        #sample batch of sequences
        sample_index = np.random.choice(args.max_length_of_trajectory, args.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :state_dim])
        #note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, state_dim:state_dim+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, state_dim+1: state_dim+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -state_dim:])

        #get q values for both state and next state but with different networks
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        #calculate q value for the update
        q_target = batch_reward + args.gamma*q_next.max(1)[0].view(args.batch_size, 1)

        
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



def main():
    net = Dqn()
    print("====================================")
    print("DQN Collection Experience...")
    print("====================================")    
    episode_reward_list = []
    for episode in range(args.max_episode):
        state = env.reset()
        net.memory_counter = 0
        ep_r=0
        step_counter = 0
        while True:
            step_counter+=1
            if args.render:
                env.render()
            action = net.choose_action(state)
            next_state, reward, done, info = env.step(action)
            #reward = reward * 100 if reward >0 else reward * 5
            ep_r += reward
            net.store_trans(state, action, reward, next_state)

            if net.memory_counter >= args.max_length_of_trajectory:
                net.learn()
                episode_reward_list.append(ep_r)
                print("Episode: \t{} | Episode Reward: \t{:0.2f} | Environment Step: {}".format(episode, ep_r, step_counter))
                break
                if done:
                    print("Episode: \t{} | Episode Reward: \t{:0.2f} | Environment Step: {}".format(episode, ep_r, step_counter))
            if done:
                print("Episode: \t{} | Episode Reward: \t{:0.2f} | Environment Step: {}".format(episode, ep_r, step_counter))
                episode_reward_list.append(ep_r)
                #if args.show_performance:
                #    net.plot(net.ax, episode_reward_list)
                break

            state = next_state
    print("====================================")
    print("DQN Ending ...")
    print("====================================")        
    if args.show_performance:
        plt.plot(range(args.max_episode),episode_reward_list,'k-')
        plt.xlabel('Episode'), plt.ylabel('Episode reward')
        plt.show()
        #plt.save_fig(directory+'reward_vs_episode.png')


if __name__ == '__main__':
    main()
