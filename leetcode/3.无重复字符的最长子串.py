#
# @lc app=leetcode.cn id=3 lang=python3
#
# [3] 无重复字符的最长子串
#

# @lc code=start
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        start, end = 0, 1
        max_length = end - start
        while end < len(s):
            if s[end] in s[start: end]:
                length = len(s[start: end])
                if length > max_length:
                    max_length = length
                start += 1
                end = start
            elif end == (len(s) - 1):
                length = len(s[start:])
                if length > max_length:
                    max_length = length
            end += 1
        return max_length
# @lc code=end

