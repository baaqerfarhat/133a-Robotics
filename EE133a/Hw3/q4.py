import rclpy
import numpy as np
from math import pi, sin, cos
from rclpy.node import Node
from sensor_msgs.msg import JointState

class Trajectory():
    def __init__(self):
        pass

    def jointnames(self):
        return ['pan', 'tilt']

    def evaluate(self, t, dt):
        theta_pan  = (pi / 3) * sin(2 * t)
        theta_tilt = (pi / 3) * sin(t) - (pi / 9) * cos(6 * t)

        omega_pan  = (2 * pi / 3) * cos(2 * t) 
        omega_tilt = (pi / 3) * cos(t) + (2 * pi / 3) * sin(6 * t) 

        return ([theta_pan, theta_tilt], [omega_pan, omega_tilt])

class Generator(Node):
    def __init__(self, name, rate):
        super().__init__(name)

        self.trajectory = Trajectory()
        self.jointnames = self.trajectory.jointnames()

        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while not self.count_subscribers('/joint_states'):
            pass

        self.dt = 1.0 / float(rate)
        self.t = -self.dt
        self.start = self.get_clock().now() + rclpy.time.Duration(seconds=self.dt)

        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" % (self.dt, rate))

    def shutdown(self):
        self.timer.destroy()
        self.destroy_node()

    def update(self):
        self.t += self.dt

        (q, qdot) = self.trajectory.evaluate(self.t, self.dt)
        now = self.start + rclpy.time.Duration(seconds=self.t)

        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg()
        cmdmsg.name = self.jointnames
        cmdmsg.position = q
        cmdmsg.velocity = qdot
        self.pub.publish(cmdmsg)

def main(args=None):
    rclpy.init(args=args)

    generator = Generator('generator', 100)
    rclpy.spin(generator)
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
