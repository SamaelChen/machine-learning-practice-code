#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for idx, n in enumerate(nums):
            residual = target - n
            if residual in nums[(idx + 1): ]:
                return(idx, nums[(idx + 1): ].index(residual) + idx + 1)
        
# @lc code=end

