#
# @lc app=leetcode.cn id=15 lang=python3
#
# [15] 三数之和
#

# @lc code=start
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        if n < 3:
            return []
        nums.sort()
        for i in range(n):
            if (i > 0 and nums[i] == nums[i-1]):
                continue
            low = i + 1
            high = n - 1
            while low < high:
                target = nums[i] + nums[low] + nums[high]
                if target == 0:
                    res.append([nums[i], nums[low], nums[high]])
                    while(low < high and nums[low] == nums[low+1]):
                        low += 1
                    while(low < high and nums[high] == nums[high-1]):
                        high -= 1
                if target < 0:
                    low += 1
                else:
                    high -= 1
        return res
# @lc code=end

