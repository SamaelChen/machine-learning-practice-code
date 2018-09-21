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
        neg = (dividend >= 0) ^ (divisor >= 0)
        dividend, divisor = abs(dividend), abs(divisor)

        pos, base = 1, divisor
        while base <= dividend:
            pos <<= 1
            base <<= 1

        base >>= 1
        pos >>= 1

        res = 0
        while pos > 0:
            if base <= dividend:
                res += pos
                dividend -= base
            base >>= 1
            pos >>= 1
        val = -res if neg else res
        return 2 ** 31 -1 if val > 2 ** 31 -1 else val


s = Solution()
%time s.divide(10, 3)
