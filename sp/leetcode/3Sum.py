"""
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0?
Find all unique triplets in the array which gives the sum of zero.

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
"""


class Solution(object):

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        A very tricky solution.
        """
        nums.sort()
        ans = []
        for l in range(len(nums) - 2):
            m = l + 1
            r = len(nums) - 1
            while m < r:
                s = nums[l] + nums[m] + nums[r]
                if s == 0:
                    ans.append([nums[l], nums[m], nums[r]])
                    m += 1
                elif s > 0:
                    r -= 1
                else:
                    m += 1
        ans = set(map(tuple, ans))
        return list(ans)


s = Solution()
s.threeSum([-1, 0, 1, 2, -1, -4])
