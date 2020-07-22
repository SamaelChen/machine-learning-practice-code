#
# @lc app=leetcode.cn id=12 lang=python3
#
# [12] 整数转罗马数字
#
# %%
# @lc code=start
class Solution:
    def intToRoman(self, num: int) -> str:
        dictionary = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM'}
        res = ''
        a = num // 1000
        num -= a * 1000
        res += 'M' * a
        b = num // 100
        num -= b * 100
        if b in dictionary:
            res += dictionary[b * 100]
        elif b < 5:
            res += 'C' * b
        else:
            res = res + 'D' + 'C' * (b - 5)
        c = num // 10
        num -= c * 10
        if c in dictionary:
            res += dictionary[c * 10]
        elif c < 5:
            res += 'X' * c
        else:
            res = res + 'L' + 'X' * (c - 5)
        d = num
        if d in dictionary:
            res += dictionary[d]
        elif d < 5:
            res += 'I' * d
        else:
            res = res + 'V' + 'I' * (d - 5)
        return res
# @lc code=end
