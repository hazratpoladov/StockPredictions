import random
from model import fflinear, Q_learn_trainer
from collections import deque
import yfinance
import torch
from plotter_things import plot


class AgentStock:
    MAX_MEMORY = 100_000
    BATCH_SIZE = 5
    LR = 0.001
    EPISODES = 2000

    def __init__(self,stock_name,period,initial_money,state_seq_size):
        #data initial money

        #trading start case
        self.profit = 0
        self.initial_money = initial_money
        self.current_money = initial_money
        self.inventory = 0 #number of stock in the hands -->>> inventory has 25 share for example

        #stock info
        self.stock_name = stock_name
        self.period = period
        self.data = None
        self.date = None


        #memory and other parameters for RL algorithms
        self.number_of_episodes = self.EPISODES
        self.memory = deque(maxlen=self.MAX_MEMORY) #defining the memory
        self.gamma = 0.95
        self.epsilon = 50
        self.state_size = state_seq_size
        self.iterations = 0
        self.action_size = 3 #buy sell and wait cases

        #model trainer done reset
        self.model = fflinear(self.state_size,8,3) #here features will be changed due to the states
        self.trainer = Q_learn_trainer(self.model,lr=self.LR,gamma=self.gamma)
        self.done = False
        self.get_data_parameters()


    def get_data_parameters(self): # there we will get the parameters of stock given name
        stock = yfinance.Ticker(self.stock_name)
        df1 = stock.history(period = self.period)
        close_df1_np = df1.Close.to_numpy()
        self.data = (close_df1_np)
        self.date = df1.index
        #print(self.data.shape)

    def get_state(self,start_index_seq):
        return self.data[start_index_seq:start_index_seq+self.state_size]


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached


    def experience_replay(self):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - self.BATCH_SIZE, l):
            mini_batch.append(self.memory[i])

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.exp_replay(states, actions, rewards, next_states, dones)




    def get_action(self,state,current_stake_value):
        self.epsilon = 10000 - self.iterations
        trade_move = [0,0,0] #reduce the dimension 0,1,2
        if random.randint(0,10000) < self.epsilon or len(self.memory) < self.BATCH_SIZE:
            move = 1
            if self.inventory == 0 and self.current_money >= current_stake_value: #here need to find the at least 1 share can be bought by using the state only
                move = random.choice([0,1])
            elif self.inventory>0 and self.current_money < current_stake_value:
                move = random.choice([1,2])
            elif self.inventory == 0 and self.current_money <=0:
                move = 1
                print("Bankrupt")
                self.done = True
            trade_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            trade_move[move] = 1
        return trade_move

    def reset(self): #it will reset the done case
        self.current_money = self.initial_money
        self.inventory = 0
        self.done = False
        self.profit = 0


    def trade_step(self): #this will trade according to the action and will return the reward, trading_over and profit
        plot_profit = []
        plot_mean_profit = []
        record_profit = 0
        total_profit = 0
        self.iterations = 0
        while(True):
            state = self.get_state(0)
            s = 0
            while(s < len(self.data) - self.state_size and not self.done):
                current_stake_value = self.data[s + self.state_size - 1]
                current_action = self.get_action(state,current_stake_value)
                next_state = self.get_state(1+s)

                if not self.done:
                    if current_action == [1,0,0] and self.current_money//self.data[s+self.state_size-1] >=1: #buy all shares possible command #
                        self.inventory = self.current_money//self.data[s+self.state_size-1]
                        self.current_money = self.current_money - self.inventory * self.data[s+self.state_size-1]
                        self.profit = self.current_money + self.inventory * self.data[s+self.state_size-1]-self.initial_money
                        print(f"{self.inventory} number of shares bought")

                    elif current_action == [0,0,1] and self.inventory >= 1:#sell command
                        self.current_money += self.inventory * self.data[s+self.state_size-1]
                        print(f"{self.inventory} number of shares sold")
                        self.inventory = 0
                        self.profit = self.current_money - self.initial_money
                    else:
                        print("wait")
                        current_action = [0, 1, 0]
                        self.profit = self.current_money + self.inventory * self.data[s + self.state_size - 1] - self.initial_money

                    if s == (len(self.data) - self.state_size-1):
                        self.done = True

                    reward = self.profit
                    print(f"profit is: {reward} in {self.date[s+self.state_size-1]}")

                    self.remember(state,current_action,reward,next_state,self.done)
                    state = next_state
                    s += 1

                if self.done:
                    self.iterations += 1
                    self.experience_replay()
                    if self.profit > record_profit:
                        record_profit = self.profit

                    plot_profit.append(self.profit)
                    total_profit += self.profit
                    mean_profit = total_profit / self.iterations
                    plot_mean_profit.append(mean_profit)
                    plot(plot_profit, plot_mean_profit)
                    print(f"record profit is: {record_profit}")
                    self.reset()


if __name__ == '__main__':

    a = AgentStock("GOOG","1000d",10000,5)
    a.trade_step()


