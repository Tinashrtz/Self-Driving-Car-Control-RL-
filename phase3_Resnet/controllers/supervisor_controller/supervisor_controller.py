from controller import Supervisor

supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())

# Get the root node
root_node = supervisor.getRoot()
# get the car node to monitor its collision 
children_node = root_node.getField("children")

car_node = children_node.getMFNode(13)
print('-'*10)

view_point_node = children_node.getMFNode(1)
# check if it has been correctly assigned a None type value
if car_node is None:
    print("Error: Node 'Linkcoln' not found in the world.")
else:
    print("Monitoring collisions for node: 'Linkcoln'")
# Initialize the receiver and emitter for transfering messages to the robot's controller and supervisor
receiver = supervisor.getDevice("receiver")
emitter = supervisor.getDevice("emitter")
receiver.enable(time_step)
receiver.setChannel(1) 
emitter.setChannel(2)
### Related to Car node
# SFVec3f
translationField = car_node.getFieldByIndex(0)
# SFRotation
OrientationField = car_node.getFieldByIndex(1)
### Related to viewpoint Node
# SFRotation
view_orientation_field = view_point_node.getFieldByIndex(1)
# SFVec3f
view_position_field = view_point_node.getFieldByIndex(2)
# initial position and orientation
startingPoseOfRobot = translationField.getSFVec3f()
startingOrienOfRobot = OrientationField.getSFRotation()
startingOrienOfView = view_orientation_field.getSFRotation()
startingPoseOfView = view_position_field.getSFVec3f()
print(startingOrienOfView)
print(startingPoseOfView)
# Simulation loop
No_Collision_happended = True # to prevent an infinit loop of reseting the simulation
while supervisor.step(time_step) != -1:

    # Get collision points for the monitored node
    contact_points = car_node.getContactPoints()
    # if the first collison has occured
    if contact_points and No_Collision_happended:
        print("Collision detected!")
        command = "COLISSION"  # Command to send
        emitter.send(command.encode("utf-8"))
        No_Collision_happended = False
        print("sent the collision message to the robot's controller")


    if receiver.getQueueLength() > 0:
        message = receiver.getString()  # Decode the received command
        receiver.nextPacket()  # Clear the packet from the queue

        # Check if the command is "RESET"
        if message == "RESET":
            print("Resetting simulation...")
            print("Initializing the environment...")
            translationField.setSFVec3f(startingPoseOfRobot)
            OrientationField.setSFRotation(startingOrienOfRobot)
            No_Collision_happended = True
            supervisor.simulationResetPhysics()
            view_orientation_field.setSFRotation(startingOrienOfView)
            view_position_field.setSFVec3f(startingPoseOfView)
