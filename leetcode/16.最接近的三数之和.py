#
# @lc app=leetcode.cn id=16 lang=python3
#
# [16] 最接近的三数之和
#

# @lc code=start
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        min_diff = 10 ** 5
        if n < 3:
            return []
        nums.sort()
        for i in range(n):
            if (i > 0 and nums[i] == nums[i-1]):
                continue
            low = i+1
            high = n-1
            while low < high:
                tmp = nums[i] + nums[low] + nums[high]
                diff = abs(tmp - target)
                if diff < min_diff:
                    res = nums[i] + nums[low] + nums[high]
                    min_diff = diff
                    while (low < high and nums[high] == nums[high-1]): high -= 1
                if tmp<target:
                    low += 1
                else:
                    high -= 1
        return res
# @lc code=end

