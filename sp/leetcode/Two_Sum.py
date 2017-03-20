'''
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
'''


class Solution1(object):

    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    res.append(i)
                    res.append(j)
        return res


class Solution2(object):

    def twoSum(self, nums, target):
        dic = {}
        for index, num in enumerate(nums):
            if num in dic:
                return [dic[num], index]
            else:
                dic[target - num] = index


s = Solution1()
s.twoSum([2, 7, 11, 15], 9)
