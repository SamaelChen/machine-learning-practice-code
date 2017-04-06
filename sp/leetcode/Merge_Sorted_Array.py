"""
Given two sorted integer arrays nums1 and nums2,
merge nums2 into nums1 as one sorted array.
You may assume that nums1 has enough space
(size that is greater or equal to m + n) to hold additional elements from nums2.
The number of elements initialized in nums1 and nums2 are m and n respectively.
"""


class Solution(object):

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        nums1[:] = nums1[:m]
        nums1.extend(nums2[:n])
        nums1.sort()


s = Solution()
a = [5, 6, 7, 8]
b = [1, 3, 4]
tmp = s.merge(a, 2, b, 2)
a
