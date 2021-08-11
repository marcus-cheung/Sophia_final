#! /usr/bin/python
from classes import Pattern_Match
import rospy
from sensor_msgs.msg import Image
from hr_msgs.msg import People
import cv2

# Used for data collection
bg = lambda s, t: s >= t
sml = lambda s, t: s <= t
comp_methods = [(cv2.HISTCMP_CORREL, bg, False),
                (cv2.HISTCMP_INTERSECT, bg, True),
                (cv2.HISTCMP_CHISQR, sml, True),
                (cv2.HISTCMP_HELLINGER, sml, False)]


def main():
    rospy.init_node('pattern_match')
    """
    Default constructor values of 0.5/0.3 thresholds for color/gray,
    10 frames required, every 3 frames, using cv2.Correl, 600 seconds until removed from memory, visualisation off)
    """
    handler = Pattern_Match()
    rospy.Subscriber("/people_test", Image, handler.image_callback)
    rospy.Subscriber('/hr/perception/people', People, handler.people_callback)
    rospy.spin()


if __name__ == '__main__':
    print('running...')
    main()