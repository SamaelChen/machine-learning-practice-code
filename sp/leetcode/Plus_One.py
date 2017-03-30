"""
Given a non-negative integer represented as a non-empty array of digits,
plus one to the integer.

You may assume the integer do not contain any leading zero,
except the number 0 itself.

The digits are stored such that the most significant digit is
at the head of the list.
"""


class Solution(object):

    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        current = sum([digits[i] * 10 ** (len(digits) - i - 1)
                       for i in range(len(digits))]) + 1
        return [int(digit) for digit in str(current)]


s = Solution()
s.plusOne([1, 2, 3, 10900])
