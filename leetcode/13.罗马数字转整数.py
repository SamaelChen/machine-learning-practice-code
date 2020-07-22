#
# @lc app=leetcode.cn id=13 lang=python3
#
# [13] 罗马数字转整数
#

# @lc code=start
class Solution:
    def romanToInt(self, s: str) -> int:
        dictionary = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = 0
        for i in range(len(s) - 1):
            if dictionary[s[i]] < dictionary[s[i + 1]]:
                flag = -1
            else:
                flag = 1
            res += dictionary[s[i]] * flag
        res += dictionary[s[-1]]
        return res
# @lc code=end

