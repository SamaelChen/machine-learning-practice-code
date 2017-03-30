"""
Given two binary strings, return their sum (also a binary string).

For example,
a = "11"
b = "1"
Return "100".
"""


class Solution(object):

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int('0b' + a, base=0) + int('0b' + b, base=0))[2:]


s = Solution()
a = '11'
b = '1'
s.addBinary(a, b)
