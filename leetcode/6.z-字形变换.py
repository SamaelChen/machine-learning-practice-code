#
# @lc app=leetcode.cn id=6 lang=python3
#
# [6] Z 字形变换
#

# @lc code=start
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows > len(s):
            return s
        res = []
        res.append([x for x in s[:numRows]])
        idx = numRows
        while idx < len(s):
            i = 1
            while i < numRows - 1 and idx < len(s):
                res.append([''] * (numRows - 1 - i) + [s[idx]] + [''] * i)
                i += 1
                idx += 1
            if i == numRows - 1:
                tmp = [x for x in s[idx:(numRows + idx)]]
                if len(tmp) < numRows:
                    tmp.extend([''] * (numRows + idx))
                res.append([x for x in tmp])
                idx += numRows
        res_s = ''
        for i in range(numRows):
            for j in range(len(res)):
                if res[j][i] != '':
                    res_s += res[j][i]
        return res_s
# @lc code=end

