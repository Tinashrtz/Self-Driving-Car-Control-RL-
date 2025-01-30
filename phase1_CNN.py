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
import tensorflow
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class DriverDevices:
    pass


driver = Driver()

time_step = int(driver.getBasicTimeStep())


def initilize_driver(driver_devices):

    # initializing the camera
    driver_devices.camera = driver.getDevice('camera')#Camera("camera")
    driver_devices.camera.enable(100)
    print(driver_devices.camera.getWidth())
    # initializing the sensors

    ## GPS sensor

    driver_devices.gps = driver.getDevice('gps')#GPS("gps")
    driver_devices.gps.enable(20)
    ## Lidar sensor

    ### Still in progress ###
    driver_devices.lidar = driver.getDevice('Sick LMS 291')#Lidar("Sick LMS 291")
    driver_devices.lidar.enable(100)
    driver_devices.lidar.enablePointCloud()
    ### Still in progress ###

    # initializing teh display
    driver_devices.display = driver.getDevice('display')
    
    # Initialize the emitter and receiver
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
    # height and width of image
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
    y = 100 # starting height of the cropping picture
    # height and width of image
    h = image.shape[0]
    w = image.shape[1]
    working_image = image[h-y:, :, :]
    # converting to hsv
    hsv_img = cv2.cvtColor(working_image,cv2.COLOR_BGR2HSV)
    # show_image(hsv_img, "ss")
    
    # perform masking
    low = [0, 0, 145]#[1, 0, 156]
    high = [50, 255, 202]#[40, 20, 255]

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
        if distance_from_main_line > 70 and distance_from_main_line < 100:
            reward = 1
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


def lane_error_detector_v2(img):
    """Implements the lane detector opencv algo its return is used for rewarding the agents actions"""
    main_contour = mainLineDetector(img)
    side_contours = sideLinesDetector(img)
    x_center = int(img.shape[1]/2)
    left_group, right_group = contour_devider(side_contours, x_center)
    M = 0
    L = 0
    R = 0
    if (len(main_contour) > 0):
        # main line detected
        M = 1
        print("Middle Line Detected")
    average_left = average_x_calculator([left_group])
    average_right = average_x_calculator([right_group])
    if average_left == 0 and average_right != 0:
        L = 0
        R = 1
        print("Only Right line Detected")
    if average_right == 0 and average_left !=0:
        R = 0
        L = 1
        print("Only Left line detected")
    # draw_cross(average_right, image.shape[0], image, (255, 255, 0))
    # draw_cross(average_left, image.shape[0], image, (255, 255, 0))
    print("lane_detector results:")
    print((L, M, R))
    return (L, M, R)




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

class WebotsEnv(Env):
    def __init__(self, driver):
        self.driver = driver
        cnn_file_path = "../small_cnn_model.h5"
        self.cnn_model = tensorflow.keras.models.load_model(cnn_file_path)
        for _ in range(50):
            self.driver.step()
        self.driver.setCruisingSpeed(20)

        # Actions we can take, steer angle 
        # 180 parts to be mapped to 180 degree in the action space
        self.action_space = Box(low=-1, high=1, shape=(1,)) # your action space
        
        # possible values for pictures
        self.cnn_output_size = (1, 150)
        self.observation_space = Box(low=0, high=1, shape=self.cnn_output_size, dtype=np.float64)

        # Set starting state a sample from the observation space
        
        self.state = np.random.uniform(low=0, high=1.0, size=(1,150))
        # self.original_state = lane_error_detector_v2(read_camera_image(driver_devices.camera.getImage()))
        # print("Original State:")
        # print(self.original_state)
        # Set Simulation length
        self.simulation_length = np.inf
        
    def step(self, action):
        """Implements the reward giving logic."""
        done = False
        self.image = read_camera_image(driver_devices.camera.getImage())
        self.state_copy = self.image[:, :, :]
        # print(np.expand_dims(self.image, axis=0).shape)
        reward = -1
        self.state = self.cnn_model.predict(np.expand_dims(self.image, axis=0))
        if len(mainLineDetector(self.state_copy)) > 0:
            reward = 1
        # reward = lane_error_detector(self.state_copy)
        # current_state = lane_error_detector_v2(self.state_copy)
        # if current_state[0] == self.original_state[0] and current_state[1] == self.original_state[1] and current_state[2] == self.original_state[2]:
        #     reward = 2
            
        new_action = action[0]
        print(new_action)
        self.driver.setSteeringAngle(new_action)
        # do some steps to see the result of the action
        SAMPLING_PERIOD = 10
        for _ in range(SAMPLING_PERIOD):
            self.driver.step()
        # Set placeholder for info
        info = {}
        if checkForCollision():
            reward = -10
            done = True
        self.render()
        print('-'*30)
        print("reward: ", reward)
        print('-'*30)
        # Return step information
        return self.state, reward, done, False, info

    def render(self):
        # Implement viz
        self.state_copy = self.image[:, :, :]
        show_image(self.state_copy)
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
        cv2.destroyAllWindows()
        reset_simulation()
        prev_speed = self.driver.getTargetCruisingSpeed()
        self.driver.setCruisingSpeed(0)
        for _ in range(20):
            self.driver.step()
        # self.original_state = lane_error_detector_v2(read_camera_image(driver_devices.camera.getImage()))
        # print("Original State:")
        # print(self.original_state)
        self.driver.setCruisingSpeed(prev_speed)
        # self.state = self.cnn_model.predict(np.expand_dims(read_camera_image(driver_devices.camera.getImage())/255, axis=0))
        self.state = self.observation_space.sample()
        return (self.state, {})


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
        # # Ensure the directory exists
        # os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        """
        This method is called at each step during training.
        """
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



# Create the custom callback to save metrics
save_datas_callback = SaveMetricsCallback(verbose=1)

env = WebotsEnv(driver)

check_env(env, warn=True)

# # Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
# Train for a specified number of timesteps
model.learn(total_timesteps=10000, callback=save_datas_callback)
# Save the trained model
model.save("../mlp_model_2_ppo_policy")
