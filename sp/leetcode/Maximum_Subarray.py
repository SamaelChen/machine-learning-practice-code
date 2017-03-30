"""
Find the contiguous subarray within an array (containing at least one number)
which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
"""


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        Kadane's algorithm
        """
        res = nums[0]
        current = nums[0]
        for num in nums[1:]:
            current = max(num, current + num)
            if current > res:
                res = current
        return res


s = Solution()
s.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
