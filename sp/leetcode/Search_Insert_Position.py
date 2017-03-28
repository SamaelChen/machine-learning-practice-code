import numpy as np
"""
Given a sorted array and a target value, return the index if the target is found.
If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Here are few examples.
[1,3,5,6], 5 → 2
[1,3,5,6], 2 → 1
[1,3,5,6], 7 → 4
[1,3,5,6], 0 → 0
"""


class Solution(object):

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums[0] > target:
            return 0
        else:
            for i in range(len(nums)):
                if nums[i] == target:
                    return i
                elif nums[i] < target:
                    flag = i + 1
            return flag


a = [1, 3, 5, 6]
s = Solution()
s.searchInsert(a, 2)
