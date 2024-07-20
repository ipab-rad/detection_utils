from math import atan2

from collections import namedtuple

Quaternion = namedtuple('Quaternion', ["x","y","z","w"])

q = Quaternion(
    0,
    0,
    -0.9073083680063325,
    -0.42046584325684006
)

yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)

print(yaw)

print(atan2(0,1))