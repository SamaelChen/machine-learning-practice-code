#
# @lc app=leetcode.cn id=8 lang=python3
#
# [8] 字符串转换整数 (atoi)
#

# @lc code=start
class Solution:
    def myAtoi(self, s: str) -> int:
        valid_ch = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        s = s.strip()
        if len(s) == 0:
            return 0
        if s[0:2] == '+-' or s[0:2] == '-+':
            return 0
        if s[0] == '+':
            s = s[1:]
        if len(s) == 0 or s[0] not in valid_ch:
            return 0
        res_raw = ''
        for idx, x in enumerate(s):
            if x in valid_ch:
                res_raw += x
            if x == '-' and idx != 0:
                break
            elif x not in valid_ch:
                break
        if res_raw[0] == '-':
            is_neg = True
        else:
            is_neg = False
        tmp = res_raw.replace('-', '')
        if len(tmp) == 0:
            return 0
        elif is_neg:
            res = -int(tmp)
        else:
            res = int(tmp)
        if res > 2** 31-1:
            return 2**31-1
        elif res < -2**31:
            return -2**31
        else:
            return res
# @lc code=end

