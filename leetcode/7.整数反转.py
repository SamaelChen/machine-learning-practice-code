#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        if x > 0:
            tmp = str(x)
            tmp = tmp[::-1]
            tmp_int = int(tmp)
        else:
            tmp = str(abs(x))
            tmp = tmp[::-1]
            tmp_int = -int(tmp)
        if tmp_int > (2**31-1) or tmp_int < -2**31:
            return 0
        else:
            return tmp_int

# @lc code=end

