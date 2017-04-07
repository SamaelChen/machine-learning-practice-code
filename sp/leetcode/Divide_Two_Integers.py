"""
Divide two integers without using multiplication, division and mod operator.

If it is overflow, return MAX_INT.
"""


class Solution(object):

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        sign = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            res += 1
            dividend -= divisor
        if not sign:
            res = -res
        return min(max(res, -2 ** 31), 2 ** 31 - 1)


s = Solution()
s.divide(-2147, -1)
