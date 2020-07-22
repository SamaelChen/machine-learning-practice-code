#
# @lc app=leetcode.cn id=18 lang=python3
#
# [18] 四数之和
#
# %%
# @lc code=start
class Solution:
    def fourSum(self, nums, target):
        n = len(nums)
        res = []
        if n < 4:
            return []
        nums.sort()
        for i in range(n-3):
            for j in range(i+1, n-2):
                if (i > 0 and nums[i] == nums[i-1]):
                    continue
                if (j > i+1 and nums[j] == nums[j-1]):
                    continue
                low = j + 1
                high = n - 1
                while low < high:
                    t = nums[i] + nums[low] + nums[high] + nums[j]
                    if t == target:
                        tmp = [nums[i], nums[j], nums[low], nums[high]]
                        tmp.sort()
                        if tmp not in res:
                            res.append(tmp)
                        while(low < high and nums[low] == nums[low+1]):
                            low += 1
                        while(low < high and nums[high] == nums[high-1]):
                            high -= 1
                    if t < target:
                        low += 1
                    else:
                        high -= 1
        return res
# @lc code=end
