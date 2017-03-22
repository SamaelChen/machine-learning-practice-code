import os
"""
Write a function to find the longest common prefix string amongst an array of strings.
"""


class Solution(object):

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        return os.path.commonprefix(strs)


s = Solution()
s.longestCommonPrefix(['abcde', 'bcd', 'bacd'])
L = ['bad', 'badsdxx', 'baxadf', 'bads']
s.longestCommonPrefix(L)
