from vehicle import Driver
from controller import Camera, GPS, Lidar, Display, Supervisor, Robot
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
# from gu.spaces import Discrete, Box
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from keras.src.legacy.saving import legacy_h5_format
import torch.nn as nn
import torch


class DriverDevices:
    pass


driver = Driver()

time_step = int(driver.getBasicTimeStep())


def initilize_driver(driver_devices):

    driver_devices.camera = driver.getDevice('camera')#Camera("camera")
    driver_devices.camera.enable(100)
    print(driver_devices.camera.getWidth())


    driver_devices.gps = driver.getDevice('gps')#GPS("gps")
    driver_devices.gps.enable(20)

    driver_devices.lidar = driver.getDevice('Sick LMS 291')#Lidar("Sick LMS 291")
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

def lidar_update():
    """Updates the LiDAR data and checks for objects on the way."""
    # Get the current LiDAR range image
    lidar_data = driver_devices.lidar.getRangeImage()
    print(f'lidar data shape = {np.array(lidar_data).shape}')

    # Convert to numpy array for easier processing
    lidar_data_array = np.array(lidar_data)
    max_range = driver_devices.lidar.getMaxRange()
    print(f'max range = {max_range}')
    lidar_data_array = np.where(np.isinf(lidar_data_array), max_range, lidar_data_array)

    # Define a threshold distance for detecting objects
    threshold_distance = 50.0  # meters

    # Filter out `inf` values and find the closest distance
    valid_distances = lidar_data_array[np.isfinite(lidar_data_array)]
    closest_distance = np.min(valid_distances) if valid_distances.size > 0 else float('inf')

    # Check if there is any object within the threshold distance
    object_detected = closest_distance < threshold_distance

    # Output results
    if object_detected:
        print(f"Object detected at distance: {closest_distance:.2f} meters")
    else:
        print("No object detected within threshold distance.")
    lidar_data_array = lidar_data_array/80
    return lidar_data_array, closest_distance


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


def average_x_calculator(contours):
    x_sum = 0
    counter = 1
    for e1 in contours:
        for e2 in e1:
            x_sum += e2[0][0]
            counter += 1
    x_avg = x_sum/counter
    return x_avg

def main_line_x_detecter(mainlinecontour):
    y_max = 0
    x_min = 0
    for contour in mainlinecontour:
        for e in contour:
            if e[0][1] > y_max:
                x_min = e[0][0]
                y_max = e[0][1]
    return x_min
def contour_devider(contours, x_treshold):
    group_1 = []
    group_2 = []
    for contour in contours:
        for e in contour:
            if e[0][0] >= x_treshold:
                group_2.append(e)
            else:
                group_1.append(e)
    return group_1, group_2
def draw_cross(x, y, image, color = (0, 255, 0)):
    x = int(x)
    y = int(y)
    cv2.line(image, (x - 10,y), (x + 10,y), color, 2)
    cv2.line(image, (x,y - 10), (x,y + 10), color, 2)
    return image


def lane_error_detector(img, preview = False, cropped_height = 50):
    """Implements the lane detector opencv algo its return is used for rewarding the agents actions"""
    main_contour = mainLineDetector(img)
    side_contours = sideLinesDetector(img)
    x_center = int(img.shape[1]/2)
    reward = -1
    if (len(main_contour) > 0):
        # main line detected
        print("main line detected")
        x_main_line = main_line_x_detecter(main_contour)
        # left_group, right_group = contour_devider(side_contours, x_main_line)
        # average_left = average_x_calculator([left_group])
        # average_right = average_x_calculator([right_group])
        # if average_left == 0:
        #     # line is on the right
        #     # print("line is on the right")
        #     x_center_right = (average_right + main_line_x)/2
        #     if np.abs(x_center_right - x_center) < 20:
        #         reward = 1
        #     # draw_cross(x_center_right, image.shape[0], image)
        # if average_right == 0:
        #     # line is on the left
        #     # print("line is on the left")
        #     x_center_left = (average_left + main_line_x)/2
        #     if np.abs(x_center_left - x_center) < 20:
        #         reward = 1
        distance_from_main_line = np.abs(x_main_line - x_center)
        if distance_from_main_line > 15 and distance_from_main_line < 50:
            reward = 4
        #     draw_cross(x_center_left, image.shape[0], image)
        # draw_cross(main_line_x, image.shape[0], image)
    else:
        left_group, right_group = contour_devider(side_contours, x_center)
        average_left = average_x_calculator([left_group])
        average_right = average_x_calculator([right_group])
        if average_left == 0 and average_right != 0:
            # line is on the right
            print("line is on the right")
            # if np.abs(average_right - x_center) < 50:
            reward = 1
        if average_right == 0 and average_left !=0:
            # line is on the left
            print("line is on the left")
            # if np.abs(average_left - x_center) < 50:
            reward = 1
        # draw_cross(average_right, image.shape[0], image, (255, 255, 0))
        # draw_cross(average_left, image.shape[0], image, (255, 255, 0))
    return reward

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

class FullyConnectedLiDAR(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedLiDAR, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)        
        self.fc3 = nn.Linear(64, 32) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class WebotsEnv(Env):
    def __init__(self, driver):
        self.driver = driver
        cnn_file_path = "small_cnn_model.h5"
        self.cnn_model = legacy_h5_format.load_model_from_hdf5(cnn_file_path, custom_objects={'mse': 'mse'})
        for _ in range(50):
            self.driver.step()
        self.driver.setCruisingSpeed(20)

        self.action_space = gym.spaces.Box(low=np.array([-0.8, 1.0]), high=np.array([0.8, 15.0]), shape=(2,),dtype=np.float32)
        
        self.cnn_output_size = (1, 150)
        self.fully_connected_model = FullyConnectedLiDAR(input_size=180)
        observation_size = (1,182)
        self.observation_space = Box(low=-0.5, high=1, shape=observation_size, dtype=np.float64)

        
        # self.state = np.random.uniform(low=0, high=1.0, size=(1,150))

        # Set Simulation length
        self.simulation_length = np.Inf
        
    def step(self, action):
        """Implements the reward giving logic."""
        done = False
        self.image = read_camera_image(driver_devices.camera.getImage())
        self.lidar_data, self.closest_distance = lidar_update()
        self.lidar_data = torch.tensor(self.lidar_data, dtype=torch.float32)
        lidar_features = self.fully_connected_model(self.lidar_data)
        lidar_features = lidar_features.detach().numpy()[:,None].T
        self.state_copy = self.image[:, :, :]
        # print(np.expand_dims(self.image, axis=0).shape)
        total_reward = -1
        image_features = self.cnn_model.predict(np.expand_dims(self.image, axis=0))
        print(f"{image_features.shape}, {lidar_features.shape}")
        self.state = np.concatenate((image_features, lidar_features), axis=1)
        print(f'state shape = {self.state.shape}')
        if len(mainLineDetector(self.state_copy)) > 0:
            total_reward = 1
        # reward = lane_error_detector(self.state_copy)
        # current_state = lane_error_detector_v2(self.state_copy)
        # if current_state[0] == self.original_state[0] and current_state[1] == self.original_state[1] and current_state[2] == self.original_state[2]:
        #     reward = 2
            
        new_action = action
        print(new_action)
        new_action[1] = (new_action[1]*10) + 10
        self.driver.setSteeringAngle(new_action[0])
        self.driver.setCruisingSpeed(new_action[1])
        # do some steps to see the result of the action
        SAMPLING_PERIOD = 10
        for _ in range(SAMPLING_PERIOD):
            self.driver.step()
        # Set placeholder for info
        info = {}
        if checkForCollision():
            total_reward = -10
            done = True
        if self.closest_distance < 5:
            total_reward -= 2
        if self.closest_distance < 2:
            total_reward -= 5
        if self.closest_distance > 40:
            total_reward += 2
        self.render()
        print('-'*30)
        print("total_reward: ", total_reward)
        print('-'*30)
        # Return step information
        return self.state, total_reward, done, False, info

    def render(self):
        # Implement viz
        self.state_copy = self.image[:, :, :]
        main_contour = mainLineDetector(self.state_copy)
        side_contours = sideLinesDetector(self.state_copy)
        # print(len(main_contour))
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


env = WebotsEnv(driver)
model = PPO.load("mlp_model_phase2_ppo_policy")

mean_reward, std_reward = evaluate_policy(model, env)
print("mean_reward")
print(mean_reward)
print("std_reward")
print(std_reward)

for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs, reward, doen, info = env.step(action)
        action, _states = model.predict(obs, deterministic=True)
        total_reward += reward
        print(f"Episode {episode + 1}: total reward = {total_reward}")