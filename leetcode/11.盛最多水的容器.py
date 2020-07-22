#
# @lc app=leetcode.cn id=11 lang=python3
#
# [11] 盛最多水的容器
#

# %%
# @lc code=start
class Solution:
    def maxArea(self, height: List[int]) -> int:
        if len(height) < 2:
            return 0
        max_area = 0
        left, right = 0, len(height) - 1
        while left != right:
            area = (right - left) * min([height[left], height[right]])
            if area > max_area:
                max_area = area
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1
        return max_area
# @lc code=end
