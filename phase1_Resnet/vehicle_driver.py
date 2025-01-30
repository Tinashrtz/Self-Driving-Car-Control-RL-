from vehicle import Driver
from controller import Camera, GPS, Lidar, Display, Supervisor, Robot
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_checker import check_env
from keras.models import load_model
from keras.src.legacy.saving import legacy_h5_format
import torch
from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import torch.nn as nn
import torchvision.models as models
import cv2
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights



class DriverDevices:
    pass


driver = Driver()

time_step = int(driver.getBasicTimeStep())


def initilize_driver(driver_devices):

    driver_devices.camera = driver.getDevice('camera')
    driver_devices.camera.enable(100)
    print(driver_devices.camera.getWidth())


    driver_devices.gps = driver.getDevice('gps')
    driver_devices.gps.enable(20)

    driver_devices.lidar = driver.getDevice('Sick LMS 291')
    driver_devices.lidar.enable(100)
    driver_devices.lidar.enablePointCloud()

    driver_devices.display = driver.getDevice('display')
    
    driver_devices.emitter = driver.getDevice("emitter")
    driver_devices.emitter.setChannel(1)  # Use channel 1 to communicate with the supervisor
    driver_devices.receiver = driver.getDevice("receiver")
    driver_devices.receiver.enable(time_step)
    driver_devices.receiver.setChannel(2) 

driver_devices = DriverDevices()
initilize_driver(driver_devices)

def read_camera_image(CameraData_getImageMethod):
        """Reads the Camera byte array and returns a cv2 image object"""
        image_array = np.frombuffer(CameraData_getImageMethod, dtype=np.uint8)
        image_array = image_array.reshape((driver_devices.camera.getHeight(), driver_devices.camera.getWidth(), 4))  # Webots returns BGRA format

        # Remove the alpha channel (last channel) for OpenCV/Matplotlib compatibility
        image_bgr = image_array[:, :, :]

        # Convert BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return image_rgb


# Line Detection Phase
def show_image(img, title="image"):
    cv2.imshow(title, img)
    cv2.waitKey(1)

def mainLineDetector(image):
    # cropping to avoid the misleading of buildings
    y = 150 # starting height of the cropping picture
    h = image.shape[0]
    w = image.shape[1]
    working_image = image[h-y:, :, :]
    # converting to hsv
    hsv_img = cv2.cvtColor(working_image,cv2.COLOR_BGR2HSV)
    # perform masking
    low = [0, 124, 79]
    high = [179, 193, 217]

    mask = cv2.inRange(hsv_img, np.array(low), np.array(high))
    hsv_gray = cv2.bitwise_and(working_image, working_image, mask=mask)
    hsv_gray = cv2.cvtColor(hsv_gray, cv2.COLOR_BGR2GRAY)

    # finding the contours
    contours, _ = cv2.findContours(hsv_gray, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE)
    # finding the largets contour
    main_line_contour = []
    if len(contours) > 0:
        new_contour = sorted(contours, key=cv2.contourArea)[-1]
        new_contour = new_contour + np.array([[[0, h-y]]], dtype=np.int32)
        main_line_contour.append(new_contour)
    return main_line_contour
    

def sideLinesDetector(image):
    # cropping to avoid the misleading of buildings
    y = 150 # starting height of the cropping picture
    h = image.shape[0]
    w = image.shape[1]
    working_image = image[h-y:, :, :]
    # converting to hsv
    hsv_img = cv2.cvtColor(working_image,cv2.COLOR_BGR2HSV)
    # show_image(hsv_img, "ss")
    
    # perform masking
    low = [1, 0, 156]
    high = [40, 20, 255]

    mask = cv2.inRange(hsv_img, np.array(low), np.array(high))
    hsv_gray = cv2.bitwise_and(working_image, working_image, mask=mask)
    hsv_gray = cv2.cvtColor(hsv_gray, cv2.COLOR_BGR2GRAY)
    
    # using contour to find the contours in the image
    contours, _ = cv2.findContours(hsv_gray, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE)
    base_img = np.zeros_like(working_image)
    for cnt in contours : 
        area = cv2.contourArea(cnt) 
    
        # Shortlisting the regions based on there area. (area of the side lines) 
        if area > 1: 
            approx = cv2.approxPolyDP(cnt,  
                                    0.009 * cv2.arcLength(cnt, True), True) 
    
            # Checking if the no. of sides of the selected region is 7. 
            cv2.drawContours(base_img, [approx], -1, color=255, thickness=cv2.FILLED) 
    
    # Apply dilation to bridge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50 , 50))  # Adjust kernel size as needed
    dilated = cv2.dilate(base_img, kernel, iterations=1)
    dilated_gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    # Find the new unified contour
    unified_contours, _ = cv2.findContours(dilated_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for contour in unified_contours:
        new_contour = contour + np.array([[[0, h-y]]], dtype=np.int32)
        new_contours.append(new_contour)
    return new_contours



def lane_error_detector(img, preview = False, cropped_height = 50):
        """Implements the lane detector opencv algo its return is used for rewarding the agents actions"""
        pass





def reset_simulation():
    print("in the reset")
    command = "RESET"  # Command to send
    driver_devices.emitter.send(command.encode("utf-8"))
    
def checkForCollision():
    # read from the supervisor channel
    if driver_devices.receiver.getQueueLength() > 0:
        message = driver_devices.receiver.getString()  # Decode the received command
        print(f"Received command: {message}")
        driver_devices.receiver.nextPacket()  # Clear the packet from the queue

        # Check if the command is "COLLISION"
        if message == "COLISSION":
            return True
    return False

def extractBoundingBox(contours):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  
        top_left = (x, y)
        top_right = (x + w, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        bounding_boxes.append([top_left, top_right, bottom_left, bottom_right])
    return bounding_boxes

import math

def calculateMinDistance(corners, point):
    min_distance = float('inf')  
    for corner in corners:
        for corner_point in corner:
            distance = math.sqrt((corner_point[0] - point[0])**2 + (corner_point[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

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

class WebotsEnv(gym.Env):
    
    def __init__(self, driver):
    
        print("in the init")
        super(WebotsEnv,self).__init__()
        self.driver = driver
        self.action_space = Box(low=-0.8, high=0.8, shape=(1,),dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,),dtype=float)
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False

        self.cnn_model = resnet18
        for _ in range(50):
            self.driver.step()
        self.driver.setCruisingSpeed(20)
        
        self.cnn_output_size = (512,1)
        self.observation_space = Box(low=0, high=1, shape=self.cnn_output_size, dtype=np.float64)
        
        self.state = np.random.uniform(low=0, high=1.0, size=(1,150))

        self.simulation_length = np.inf
        
    def step(self, action):
        """Implements the reward giving logic."""
        self.image = read_camera_image(driver_devices.camera.getImage())
        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        print(f'({image_width}, {image_height})')
        image_center = (int(image_width/2), image_height)
        print(image_center)
        resized_image = cv2.resize(self.image, dsize=(224, 224))  # Fix the size to 224x224
        self.state = feature_extractor(model=self.cnn_model, image=resized_image)
        print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ state shape = {self.state.shape}')
        # computing the reward
        total_reward = -100
        self.state_copy = self.image[:, :, :]
        main_line_countour = mainLineDetector(self.state_copy)
        main_contour_x = None
        y_max = 0
        print(np.array(main_line_countour).shape)
        for e1 in main_line_countour:
            for pixel in e1:
                if pixel[0][1]>y_max:
                    y_max = pixel[0][1]
                    main_contour_x = pixel[0][0]
        print(main_contour_x)
        main_line_position = (main_contour_x, y_max)
        try:
            blue_distance = math.sqrt((main_line_position[0] - image_center[0])**2 + (main_line_position[1] - image_center[1])**2)
        except:
            blue_distance = None
        print(f'blue distance = {blue_distance}')
        side_lines_contours = sideLinesDetector(self.state_copy)
        bounding_boxes = extractBoundingBox(side_lines_contours)
        print(bounding_boxes)
        try:
            red_distance = calculateMinDistance(corners=bounding_boxes, point=image_center)
        except:
            red_distance = None
        print(f'red distance = {red_distance}')

        # print(np.array(side_lines_contours).shape)
        if len(mainLineDetector(self.state_copy)) > 0:
            total_reward += 2
        new_action = action[0]
        self.driver.setSteeringAngle(new_action)
        # do some steps to see the result of the action
        SAMPLING_PERIOD = 10
        for _ in range(SAMPLING_PERIOD):
            self.driver.step()
        # Set placeholder for info
        info = {}
        if checkForCollision():
            total_reward += -1000
            done = True
        else:
            done = False
            if red_distance != None and red_distance<20:
                total_reward += -500
                done = True
            elif blue_distance != None and blue_distance<20:
                total_reward += 500
            else:
                if red_distance!=None and blue_distance!=None:
                    total_reward += 1/(np.abs(blue_distance - red_distance)) * 100000
            if red_distance!=None and blue_distance==None:
                total_reward += -350
            
        # gray_version = cv2.cvtColor(self.state, cv2.COLOR_BGR2GRAY)
        # # if we are seeing a total black color we don't send it but instead we send a white color to represent 
        # if cv2.countNonZero(gray_version) == 0:
        #     print("Image is black")
        #     self.state = self.observation_space.sample()
        # else:
        #     pass
        # reward = 1
        self.render()
        # self.state = self.state /255.0
        # Return step information
        print(f'total reward = {total_reward}')
        return self.state, total_reward, done, False, info

    def render(self):
        # Implement visualization
        self.state_copy = self.image[:, :, :]
        main_contour = mainLineDetector(self.state_copy)
        side_contours = sideLinesDetector(self.state_copy)
        print(len(main_contour))
        # print(len(side_contours))
        cv2.drawContours(self.state_copy, main_contour, -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(self.state_copy, side_contours, -1, (0, 0, 255), cv2.FILLED)
        show_image(self.state_copy)

    def reset(self, seed = None, options= None):
        """Resets the environment to its original state and returns a sample state from the observation space."""
        super().reset(seed=seed, options=options)
        self.state = self.observation_space.sample()
        cv2.destroyAllWindows()
        reset_simulation()
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
    

env = DummyVecEnv([lambda: WebotsEnv(driver)])

save_path = os.path.join(os.getcwd(), "ppo_model.zip")

if os.path.exists(save_path):
    print("Model file found. Loading existing model...")
    model = PPO.load(save_path, env=env)
    print("Model loaded successfully.")
else:
    print("No model file found. Creating a new model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=50)

print("Training the model...")
model.learn(total_timesteps=100)

model.save(save_path)
print(f"Model saved at {save_path}")

obs = env.reset()
done = False
step_count = 0
max_steps = 500

print("Starting interaction with the environment...")

while step_count < max_steps:
    print(f"Step {step_count + 1}:")
    
    action, _states = model.predict(obs, deterministic=True)
    print(f"Predicted action: {action}")

    obs, rewards, dones, infos = env.step(action)
    print(f"Rewards: {rewards}, Dones: {dones}")

    if dones[0]:  
        print("Episode done. Resetting environment manually...")
        obs = env.reset()  
        done = False 

    step_count += 1

    if step_count % 10 == 0:
        try:
            env.render()
        except NotImplementedError:
            print("Render not supported by the environment.")

print("Simulation completed.")
env.close()