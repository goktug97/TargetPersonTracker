import cv2
import rospy
from sensor_msgs.msg import Image

if __name__ == '__main__':
    from options import args
    cap = cv2.VideoCapture(args.input)
    # Camera Settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if (args.camera_fps):
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    # Init ROS Node
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    rospy.init_node('camera', anonymous=True)

    i = 0
    while not rospy.is_shutdown():
        msg = Image()
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = frame.size
        rows, cols, c = frame.shape
        msg.height = rows
        msg.width = cols
        msg.encoding = 'bgr8'
        msg.step = size//rows
        msg.data = list(frame.reshape(size))
        msg.header.frame_id = str(i)
        image_pub.publish(msg)
        i += 1
