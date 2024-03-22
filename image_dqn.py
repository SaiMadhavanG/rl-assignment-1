import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import deque
import random
from tqdm.auto import tqdm
import pickle
import os
from datetime import datetime
import json
import sys


device = "cuda" if torch.cuda.is_available() else "cpu"


def createDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


resize = torchvision.transforms.Resize(224)


def processImg(img):
    img = img / 255
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor(img, dtype=torch.float32)
    img = resize(img)
    return img


class DQN:
    def __init__(
        self,
        env,
        gamma,
        epsilon,
        numEpisodes,
        stateDim,
        actionDim,
        replayBufferSize,
        batchSize,
        epsilonDecay,
        epsilonDecayFrequency,
        lr,
        seqLen,
        checkpointDir="checkpoints",
        checkpointFrequency=5000,
        runId=datetime.now().strftime("%y_%m_%d__%H_%M_%S"),
        inferenceFrequeny=10,
        terminationSteps=500,
        haltConditionChain=2,
        loadCheckpoint=None,
        writer=False,
        imageModel="./models/best_model_params.pt",
    ):
        # Initialize DQN parameters and variables
        self.env = gym.make(env, render_mode="rgb_array")  # Environment
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.numEpisodes = numEpisodes  # Number of training episodes
        self.stateDim = stateDim * seqLen  # Dimensionality of state space
        self.actionDim = actionDim  # Dimensionality of action space
        self.replayBufferSize = replayBufferSize  # Size of the replay buffer
        self.batchSize = batchSize  # Mini-batch size for training
        self.epsilonDecay = epsilonDecay  # Rate of exploration decay
        self.epsilonDecayFrequency = (
            epsilonDecayFrequency  # Episode at which to start epsilon decay
        )
        self.checkpointDir = checkpointDir  # Folder for storing checkpoints
        self.checkpointFrequency = (
            checkpointFrequency  # Frequency (in steps) for storing checkpoints
        )
        self.runId = runId  # A run id to uniquely characterize every run / train attempt. By default takes current time as value
        self.inferenceFrequency = inferenceFrequeny  # Period for running inference
        self.writer = SummaryWriter("runs/" + self.runId) if writer else None
        self.terminationSteps = terminationSteps
        self.terminated = False
        self.haltConditionChain = haltConditionChain
        self.haltChain = 0
        self.seqLen = seqLen
        self.imageModel = torchvision.models.resnet18()
        self.imageModel.fc = nn.Linear(512, 2)
        self.imageModel.load_state_dict(torch.load(imageModel))
        self.imageModel = self.imageModel.to(device)

        # Initialize counters and lists
        self.steps = 0
        self.episodes = 0
        self.sumRewardsEpisode = []  # Track sum of rewards per episode
        self.inferenceRewards = []

        # Initialize replay buffer
        self.replayBuffer = deque(maxlen=self.replayBufferSize)

        # Initialize neural network
        self.network = self.createNetwork(self.stateDim, self.actionDim)
        # if loadCheckpoint:
        #     self = DQN.load_checkpoint(loadCheckpoint)

        # Initialize actions history
        self.actionsHistory = []

        # Initialize optimizer
        self.optim = torch.optim.Adam(self.network.parameters(), lr=lr)

    # Define the neural network architecture
    def createNetwork(self, inDim, outDim):
        return nn.Sequential(
            nn.Linear(inDim, 64),  # Input layer with ReLU activation
            nn.ReLU(),
            nn.Linear(64, 64),  # Hidden layer with ReLU activation
            nn.ReLU(),
            nn.Linear(64, 32),  # Another hidden layer with ReLU activation
            nn.ReLU(),
            nn.Linear(32, outDim),  # Output layer
        ).to(device)

    # Define custom loss function for DQN
    def my_loss(self, y_pred, y_true):
        # Calculate mean squared error loss only for the actions taken
        s1, s2 = y_true.shape
        loss = F.mse_loss(
            y_pred[torch.arange(s1), self.actionsHistory],
            y_true[torch.arange(s1), self.actionsHistory],
        )
        return loss

    # Train the network over multiple episodes
    def trainingEpisodes(self):
        # Initialize lists to store episode rewards and average Q values
        self.sumRewardsEpisode = []
        self.avgQ = []
        # Iterate over episodes
        for episode in tqdm(range(self.numEpisodes)):
            rewards = []
            # Reset environment and get initial state
            self.env.reset()
            terminalState = False
            self.Qs = []  # List to store Q values
            numSteps = 0
            states = [self.getState()] * self.seqLen
            currentState = torch.concat(states)

            # Loop until episode termination
            while not terminalState and numSteps < self.terminationSteps:

                # Select action using epsilon-greedy policy
                action = self.selectAction(currentState, episode)
                # Take action and observe next state and reward
                (_, reward, terminalState, _, _) = self.env.step(action)
                rewards.append(reward)
                states.append(self.getState())
                states.pop(0)
                nextState = torch.concat(states)
                # Store experience in replay buffer
                self.replayBuffer.append(
                    (currentState, action, reward, nextState, terminalState)
                )
                # Train the network
                self.trainNetwork()
                currentState = nextState
                numSteps += 1

            # Store episode reward and average Q value
            self.sumRewardsEpisode.append(np.sum(rewards))
            self.avgQ.append(np.mean(self.Qs))

            # Updating tensor board
            if self.writer:
                self.writer.add_scalar("steps", self.steps, self.episodes)
                self.writer.add_scalar(
                    "rewards", self.sumRewardsEpisode[self.episodes], self.episodes
                )
                self.writer.add_scalar(
                    "avg Q value", self.avgQ[self.episodes], self.episodes
                )
                self.writer.add_scalar("epsilon", self.epsilon, self.episodes)

            self.episodes += 1

            if self.episodes % self.inferenceFrequency == 0:
                self.runInference()
                if self.writer:
                    self.writer.add_scalar(
                        "inference", self.inferenceRewards[-1], self.episodes
                    )

            if self.terminated:
                print("Terminated!")
                self.save_checkpoint()
                return

    def getState(self):
        img = processImg(self.env.render()).to(device)
        img = torch.unsqueeze(img, 0)
        return torch.squeeze(self.imageModel(img))

    # Select action using epsilon-greedy policy
    def selectAction(self, state, index):
        self.network.eval()
        # Initial exploration phase
        if index == 0:
            return np.random.choice(self.actionDim)

        rand = np.random.random()
        # Decay exploration rate over time
        if self.steps % self.epsilonDecayFrequency == 0 and self.steps:
            self.epsilon *= self.epsilonDecay
            print(f"Steps: {self.steps}, Epsilon: {self.epsilon}")

        if rand < self.epsilon:
            return np.random.choice(self.actionDim)  # Random action
        # Exploit learned policy
        Q = self.network(state.reshape(1, self.stateDim).to(device))
        return torch.argmax(Q[0]).item()

    # Train the network using experiences from replay buffer
    def trainNetwork(self):
        if len(self.replayBuffer) > self.batchSize:
            # Sample a mini-batch from replay buffer
            randomBatch = random.sample(self.replayBuffer, self.batchSize)
            currentStateBatch = torch.zeros(size=(self.batchSize, self.stateDim)).to(
                device
            )
            nextStateBatch = torch.zeros(size=(self.batchSize, self.stateDim)).to(
                device
            )

            for index, tupleS in enumerate(randomBatch):
                # Extract data from the mini-batch
                currentStateBatch[index, :] = tupleS[0]
                nextStateBatch[index, :] = tupleS[3]

            # Set network to evaluation mode
            self.network.eval()
            QS_ = self.network(nextStateBatch)  # Compute Q-values for next states

            Y = torch.zeros((self.batchSize, 2)).to(device)
            self.actionsHistory = []
            rewards_batch = []
            for idx, (currentState, action, reward, nextState, terminated) in enumerate(
                randomBatch
            ):
                rewards_batch.append(reward)
                if terminated:
                    y = reward
                else:
                    # Calculate target Q-value using target network
                    y = reward + self.gamma * torch.max(QS_[idx])
                self.actionsHistory.append(action)
                Y[idx, action] = y

            # Set network back to training mode
            self.network.train()

            # Zero out gradients
            self.optim.zero_grad()

            # Compute Q-values for current states
            QS = self.network(currentStateBatch).to(device)

            # Compute loss and backpropagate
            loss = self.my_loss(QS, Y)
            loss.backward(retain_graph=True)
            self.optim.step()  # Update network parameters

            self.Qs.append(QS.flatten().sum().item())  # Track sum of Q-values

            self.steps += 1  # Increment step counter

            # Saving checkpoints periodically
            if self.steps % self.checkpointFrequency == 0:
                self.save_checkpoint()

    def runInference(self):
        self.network.eval()

        # Initialize lists to store episode rewards and average Q values
        # Iterate over episodes

        rewards = []
        # Reset environment and get initial state
        (_, _) = self.env.reset()
        terminalState = False
        self.Qs = []  # List to store Q values
        numSteps = 0
        states = [self.getState()] * self.seqLen
        currentState = torch.concat(states)
        # Loop until episode termination
        while not terminalState and numSteps < self.terminationSteps:
            # Select action using epsilon-greedy policy
            Q = self.network(currentState.reshape(1, self.stateDim).to(device))
            action = torch.argmax(Q[0]).item()
            # Take action and observe next state and reward
            (_, reward, terminalState, _, _) = self.env.step(action)
            states.append(self.getState())
            states.pop(0)
            nextState = torch.concat(states)
            rewards.append(reward)
            currentState = nextState
            numSteps += 1

        # Store episode reward and average Q value
        self.inferenceRewards.append(np.sum(rewards))

        if numSteps == self.terminationSteps:
            self.haltChain += 1
        else:
            self.haltChain = 0

        if self.haltChain == self.haltConditionChain:
            self.terminated = True

    # A function to save checkpoints
    def save_checkpoint(self):
        # !TODO self.writer and self.env is not picklable. this is a workaround I am not too happy with
        tempWriter = self.writer
        self.writer = None
        tempEnv = self.env
        self.env = None

        createDir(self.checkpointDir)
        createDir(os.path.join(self.checkpointDir, self.runId))
        checkpointPath = os.path.join(
            self.checkpointDir, self.runId, f"checkpoint-{self.steps}.pkl"
        )
        with open(checkpointPath, "wb") as f:
            pickle.dump(self, f)

        self.writer = tempWriter
        self.env = tempEnv

    # A class method to load a checkpoint
    @classmethod
    def load_checkpoint(cls, checkpointPath):
        with open(checkpointPath, "rb") as f:
            dqn = pickle.load(f)
        return dqn


if __name__ == "__main__":
    params = json.load(open(sys.argv[1]))
    if sys.argv[2] == "train":
        dqn = DQN(**params)
        dqn.trainingEpisodes()
    elif sys.argv[2] == "inference":
        N = int(sys.argv[3])
        dqn = DQN.load_checkpoint(params["loadCheckpoint"])
        dqn.env = gym.make(params["env"], render_mode="human")
        s = 0
        for i in range(N):
            dqn.runInference()
            print(dqn.inferenceRewards[-1])
            s += dqn.inferenceRewards[-1]
        print(f"Average score: {s/N:.2f}")
