#
# @lc app=leetcode.cn id=5 lang=python3
#
# [5] 最长回文子串
#

# @lc code=start
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1:
            return s
        res = s[0]
        max_length = 0
        for start in range(len(s) - 1):
            end = start + 1
            while end < len(s):
                if s[end] == s[start]:
                    if max_length < len(s[start:(end + 1)]) and s[start: (end + 1)] == s[start: (end+1)][::-1]:
                        res = s[start: (end+1)]
                        max_length = len(res)
                end += 1
        return res

# @lc code=end

