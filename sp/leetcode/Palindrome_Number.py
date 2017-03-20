"""
Determine whether an integer is a palindrome. Do this without extra space.
"""
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        else:
            y = int(str(x)[::-1])
        return y == x


s = Solution()
s.isPalindrome(-2147447412)
