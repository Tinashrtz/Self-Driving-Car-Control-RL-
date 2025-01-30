# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver controller."""

import cv2
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
# from gu.spaces import Discrete, Box
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_checker import check_env
from vehicle import Driver
from controller import Camera, GPS, Lidar, Display, Supervisor, Robot
from keras.models import load_model
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import torch



sensorsNames = [
    "front",
    "front right 0",
    "front right 1",
    "front right 2",
    "front left 0",
    "front left 1",
    "front left 2",
    "rear",
    "rear left",
    "rear right",
    "right",
    "left"]
sensors = {}

lanePositions = [10.6, 6.875, 3.2]
currentLane = 1
overtakingSide = None
maxSpeed = 80
safeOvertake = False

# supervisor = Supervisor()
# supervisor.simulationReset()
# supervisor.simulationResetPhysics()

def get_filtered_speed(speed):
    """Filter the speed command to avoid abrupt speed changes."""
    get_filtered_speed.previousSpeeds.append(speed)
    if len(get_filtered_speed.previousSpeeds) > 100:  # keep only 80 values
        get_filtered_speed.previousSpeeds.pop(0)
    return sum(get_filtered_speed.previousSpeeds) / float(len(get_filtered_speed.previousSpeeds))


def is_vehicle_on_side(side):
    """Check (using the 3 appropriated front distance sensors) if there is a car in front."""
    for i in range(3):
        name = "front " + side + " " + str(i)
        if sensors[name].getValue() > 0.8 * sensors[name].getMaxValue():
            return True
    return False


def reduce_speed_if_vehicle_on_side(speed, side):
    """Reduce the speed if there is some vehicle on the side given in argument."""
    minRatio = 1
    for i in range(3):
        name = "front " + overtakingSide + " " + str(i)
        ratio = sensors[name].getValue() / sensors[name].getMaxValue()
        if ratio < minRatio:
            minRatio = ratio
    return minRatio * speed

def distance_getValue(sensors):
    sensor_get_value = []
    print(sensorsNames)
    for name in sensorsNames:
        print(name)
        sensor_get_value.append(sensors[name].getValue())
    
    return sensor_get_value

driver = Driver()
get_filtered_speed.previousSpeeds = []
for name in sensorsNames:
    sensors[name] = driver.getDevice("distance sensor " + name)
    sensors[name].enable(10)

gps = driver.getDevice("gps")
gps.enable(10)

camera = driver.getDevice("camera")
# uncomment those lines to enable the camera
camera.enable(10)
camera.recognitionEnable(50)

lidar_right = driver.getDevice("ibeo 1")
lidar_right.enable(10)
lidar_left = driver.getDevice("ibeo 2")
lidar_left.enable(10)


def read_camera_image(CameraData_getImageMethod):
        """Reads the Camera byte array and returns a cv2 image object"""
        image_array = np.frombuffer(CameraData_getImageMethod, dtype=np.uint8)
        image_array = image_array.reshape((camera.getHeight(), camera.getWidth(), 4))  # Webots returns BGRA format

        # Remove the alpha channel (last channel) for OpenCV/Matplotlib compatibility
        image_bgr = image_array[:, :, :]

        # Convert BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_rgb


# Line Detection Phase
def show_image(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(1)

def lidar_update():
    """Updates the LiDAR data and checks for objects on the way."""
    # Get the current LiDAR range image
    lidar_right_data = lidar_right.getRangeImage()
    print(f'lidar data shape = {np.array(lidar_right_data).shape}')

    lidar_left_data = lidar_left.getRangeImage()
    print(f'lidar data shape = {np.array(lidar_left_data).shape}')

    # Convert to numpy array for easier processing
    lidar_right_data = np.array(lidar_right_data)
    max_range = lidar_right.getMaxRange()
    print(f'max range = {max_range}')
    lidar_right_data = np.where(np.isinf(lidar_right_data), max_range, lidar_right_data)

    # Convert to numpy array for easier processing
    lidar_left_data = np.array(lidar_left_data)
    max_range = lidar_left.getMaxRange()
    print(f'max range = {max_range}')
    lidar_left_data = np.where(np.isinf(lidar_left_data), max_range, lidar_left_data)

    min_distance_right = np.min(lidar_right_data)
    min_distance_left = np.min(lidar_left_data)

    lidar_right_data = lidar_right_data/200
    lidar_left_data = lidar_left_data/200
    return lidar_right_data, lidar_left_data, min_distance_right, min_distance_left

def feature_extractor(model, image:np.ndarray):
    tensor_image = torch.tensor(image)  # Image is in H x W x C format
    tensor_image = tensor_image.permute(2, 0, 1)  # Change to C x H x W format
    tensor_image = tensor_image.float() / 255.0   # Normalize to [0, 1]

    print("Tensor shape after conversion:", tensor_image.shape)

    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)

    # Define the normalization transform (ImageNet mean and std)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Apply normalization
    tensor_image = normalize(tensor_image.squeeze(0)).unsqueeze(0)

    # Print the tensor shape to verify
    print("Final tensor shape:", tensor_image.shape)  # Should output torch.Size([1, 3, 224, 224])

    features_var = model(tensor_image) # get the output from the last hidden layer of the pretrained resnet
    features_var = features_var[0, :, 0, 0][:,None]
    print("Output shape:", features_var.shape)
    features_var = features_var.numpy()
    return features_var

class FullyConnectedLiDAR(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedLiDAR, self).__init__()
        self.fc1 = nn.Linear(input_size, 2500)  # لایه اول
        self.fc2 = nn.Linear(2500, 1000)         # لایه دوم
        self.fc3 = nn.Linear(1000, 500) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
for name in sensorsNames:
    sensors[name] = driver.getDevice("distance sensor " + name)
    sensors[name].enable(10)

gps = driver.getDevice("gps")
gps.enable(10)

camera = driver.getDevice("camera")
# uncomment those lines to enable the camera
camera.enable(10)
camera.recognitionEnable(50)

time_step = int(driver.getBasicTimeStep())

# Initialize the emitter and receiver
emitter = driver.getDevice("emitter")
emitter.setChannel(1)  # Use channel 1 to communicate with the supervisor
receiver = driver.getDevice("receiver")
receiver.enable(time_step)
receiver.setChannel(2) 

def read_camera_image(CameraData_getImageMethod):
        """Reads the Camera byte array and returns a cv2 image object"""
        image_array = np.frombuffer(CameraData_getImageMethod, dtype=np.uint8)
        image_array = image_array.reshape((camera.getHeight(), camera.getWidth(), 4))  # Webots returns BGRA format

        # Remove the alpha channel (last channel) for OpenCV/Matplotlib compatibility
        image_bgr = image_array[:, :, :]

        # Convert BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_rgb


# Line Detection Phase
def show_image(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(1)

def distance_getValue(sensors):
    sensor_get_value = []
    for name in sensorsNames:
        sensor_get_value.append(sensors[name].getValue())
    
    return sensor_get_value


def reset_simulation():
    print("in the reset")
    command = "RESET"  # Command to send
    emitter.send(command.encode("utf-8"))
    
def checkForCollision():
    # read from the supervisor channel
    if receiver.getQueueLength() > 0:
        message = receiver.getString()  # Decode the received command
        print(f"Received command: {message}")
        receiver.nextPacket()  # Clear the packet from the queue

        # Check if the command is "COLLISION"
        if message == "COLISSION":
            return True
    return False

class WebotsEnv(gym.Env):
    
    def __init__(self, driver):
    
        super(WebotsEnv,self).__init__()
        self.driver = driver
        self.action_space = gym.spaces.Box(low=np.array([-0.8, 10]), high=np.array([0.8, 80]), shape=(2,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,),dtype=float)
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False

        self.cnn_model = resnet18
        for _ in range(50):
            self.driver.step()
        self.driver.setCruisingSpeed(10)
        
        # possible values for pictures
        self.cnn_output_size = (512,1)
        observation_size = (1012,1)
        self.observation_space = Box(low=0, high=1, shape=observation_size, dtype=np.float64)



        # Set Simulation length
        self.simulation_length = np.inf

    def step(self, action):
        """Implements the reward giving logic."""
        self.image = read_camera_image(camera.getImage())
        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        # print(f'({image_width}, {image_height})')
        image_center = (int(image_width/2), image_height)
        # print(image_center)

        image_features = feature_extractor(self.cnn_model, self.image)
        # print(image_features.shape)
        fully_connected = FullyConnectedLiDAR(input_size=5440)
        lidar_right_data, lidar_left_data, min_distance_right, min_distance_left = lidar_update()
        lidar_data = np.concatenate((lidar_left_data, lidar_right_data), axis=0)
        lidar_data = torch.tensor(lidar_data, dtype=torch.float32)
        lidar_features = fully_connected(lidar_data)
        lidar_features = lidar_features.detach().numpy()[:,None]
        self.state = np.concatenate((image_features, lidar_features), axis=0)
        # print(f'state shape = {self.state.shape}')
        # computing the reward
        done = False
        total_reward = -1
        distances = distance_getValue(sensors)
        distances = distances[0:7] + distances[10:]
        mean_right = np.mean(np.array([distances[1:4]]))
        mean_left = np.mean(np.array([distances[4:7]]))
        distances = [distances[0], mean_left, mean_right, distances[7], distances[8]]
        car_position = gps.getValues()
        car_x = car_position[1]
        # lanePositions = [10.6, 6.875, 3.2]
        # reward calculation
        if (np.abs(car_x - 3.2) < 0.5) or (np.abs(car_x - 6.875) < 0.5) or (np.abs(car_x - 10.6) < 0.5):
            total_reward += 10
        if (np.abs(car_x - 5) < 0.1) or (np.abs(car_x - 8.7) < 0.1):
            total_reward -= 10
        if (distances[3]<0.8) or (distances[4]<0.8):
            total_reward -= 20
        if (distances[0]< 8.0) or (mean_left< 8.0) or (mean_right< 8.0):
            total_reward -= 30

        new_action = action
        print(f'new action shape = {new_action}')
        self.driver.setSteeringAngle(new_action[0])
        self.driver.setCruisingSpeed(new_action[1])
        # do some steps to see the result of the action
        SAMPLING_PERIOD = 10
        for _ in range(SAMPLING_PERIOD):
            self.driver.step()
        # Set placeholder for info
        info = {}
        if checkForCollision():
            done = True
            total_reward -= 100
        # print(f'total_reward = {total_reward}')
        return self.state, total_reward, done, False, info
    def reset(self, *, seed = None, options = None):
        """Resets the environment to its original state and returns a sample state from the observation space."""
        super().reset(seed=seed, options=options)
        cv2.destroyAllWindows()
        reset_simulation()
        prev_speed = self.driver.getTargetCruisingSpeed()
        self.driver.setCruisingSpeed(0)
        for _ in range(20):
            self.driver.step()
        self.driver.setCruisingSpeed(prev_speed)
        self.state = self.observation_space.sample()
        return (self.state, {})

class LossLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossLoggingCallback, self).__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        # Extract the policy and value losses from the model's `logger`
        logs = self.locals["infos"]
        for info in logs:
            if "policy_loss" in info and "value_loss" in info:
                self.losses.append((info["policy_loss"], info["value_loss"]))
                if self.verbose > 0:
                    print(f"Policy Loss: {info['policy_loss']}, Value Loss: {info['value_loss']}")
        return True

class SaveMetricsCallback(BaseCallback):
    """
    A custom callback to save episode rewards and lengths during training to a file.
    """
    def __init__(self, verbose=0):
        super(SaveMetricsCallback, self).__init__(verbose)
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": []
        }
        self.loss_values = []

    def _on_step(self) -> bool:
        """
        This method is called at each step during training.
        """
        print("Keys in self.locals:", self.locals.keys())
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")
        loss = self.locals.get("loss")
        if reward is not None:
            self.current_episode_reward += reward
            self.current_episode_length += 1
            print("total reward:", self.current_episode_reward[0])
            print("episode length:", self.current_episode_length)
        if "train/loss" in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value["train/loss"]
            self.loss_values.append(loss)
            print("loss:", loss)

        if done and np.any(done):  # Check if the episode is done
            
            self.metrics["episode_rewards"].append(self.current_episode_reward[0])
            self.metrics["episode_lengths"].append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_start(self):
        """
        This method is called at the start of training.
        """
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_training_end(self):
        """
        This method is called at the end of training.
        """
        # Save metrics to the specified file
        rewards = np.array(self.metrics["episode_rewards"])
        episode_length = np.array(self.metrics["episode_lengths"])
        loss_array = np.array(self.loss_values)
        np.save("rewards.npy", rewards)
        np.save("episods.npy", episode_length)
        np.save("loss.npy", loss_array)
        print("Metrics saved: rewards.npy, episods.npy, loss.npy")

class ResetCallback(BaseCallback):
    def __init__(self, supervisor: Supervisor, reset_interval: int, verbose=0):
        super().__init__(verbose)
        self.supervisor = supervisor
        self.reset_interval = reset_interval

    def _on_step(self) -> bool:
        if self.num_timesteps % self.reset_interval == 0:
            self.supervisor.simulationReset()
            self.supervisor.simulationResetPhysics()
            print(f"Simulation reset at timestep {self.num_timesteps}.")
        return True
    

env = DummyVecEnv([lambda: WebotsEnv(driver)])

# Define the save path
save_path = "ppo_model1000.zip"

# Check if the model file exists
if os.path.exists(save_path):
    print("Model file found. Loading existing model...")
    model = PPO.load(save_path, env=env)
    print("Model loaded successfully.")
else:
    print("No model file found. Creating a new model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=50)

save_datas_callback = SaveMetricsCallback(verbose=1)

# Train or continue training the model
print("Training the model...")
supervisor = Supervisor()

reset_interval = 100 
callback = ResetCallback(supervisor=supervisor, reset_interval=reset_interval)

model.learn(total_timesteps=1000, callback=[callback, save_datas_callback])

model.save(save_path)