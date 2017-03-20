"""
Reverse digits of an integer.

Example1: x = 123, return 321
Example2: x = -123, return -321
NOTE: The input is assumed to be a 32-bit signed integer.
Your function should return 0 when the reversed integer overflows.
"""


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if(x < 0):
            y = 0 - int(str(abs(x))[::-1])
        else:
            y = int(str(x)[::-1])
        if y > 2 ** 31 or y < -2 ** 31:
            return 0
        else:
            return y


s = Solution()
s.reverse(-2147483648)
