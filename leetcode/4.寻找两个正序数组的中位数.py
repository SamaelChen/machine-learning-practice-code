#
# @lc app=leetcode.cn id=4 lang=python3
#
# [4] 寻找两个正序数组的中位数
#

# @lc code=start
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        l = []
        l.extend(nums1)
        l.extend(nums2)
        l.sort()
        if len(l) % 2 == 0:
            return (l[int(len(l) / 2)] + l[int(len(l)/ 2 - 1)]) / 2
        else:
            return l[len(l) // 2]
# @lc code=end

