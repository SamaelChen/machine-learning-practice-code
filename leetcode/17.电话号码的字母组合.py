#
# @lc app=leetcode.cn id=17 lang=python3
#
# [17] 电话号码的字母组合
#

# @lc code=start
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        digit2ch = {'2': ['a', 'b', 'c'],
                    '3': ['d', 'e', 'f'],
                    '4': ['g', 'h', 'i'],
                    '5': ['j', 'k', 'l'],
                    '6': ['m', 'n', 'o'],
                    '7': ['p', 'q', 'r', 's'],
                    '8': ['t', 'u', 'v'],
                    '9': ['w', 'x', 'y', 'z']
                    }
        if not digits: return []
        res = []
        for d in digits:
            tmp = []
            if len(res) == 0:
                res = digit2ch[d]
            else:
                for a in res:
                    for b in digit2ch[d]:
                        tmp.append(a + b)
                res = tmp
        return res
# @lc code=end

