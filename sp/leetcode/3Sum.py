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
        A very tricky solution. But it costs time, not a fastest solution.
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

    def threeSum2(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        This is a fastest solution, but it is really hard to understand.
        """
        # it's pointless to have more than two instances of any number other than
        # 0, and pointless to have more than three instances of 0. Simplification:
        # delete extraneous instances
        instances = {}
        for n in nums:
            if n in instances:
                count = instances[n]
                if count == 1 or (n == 0 and count == 2):
                    instances[n] += 1
            else:
                instances[n] = 1

        # remove extraneous duplicate values. Three 0's is always useful, but two
        # 0's isn't because no third value sums with them to 0. When count = 3 the
        # value must be 0, so leave that alone but otherwise 0 gets no exception.
        # For other values n, count = 2 is only useful when the value -2n is
        # available.
        for n, count in instances.iteritems():
            if count == 2 and (n == 0 or -2 * n not in instances):
                instances[n] = 1

        # create an ordered list of values
        values = []
        for n, count in sorted(instances.iteritems()):
            for i in range(count):
                values.append(n)
        nvalues = len(values)
        while nvalues >= 4:
            floor = -(values[nvalues - 1] + values[nvalues - 2])
            ceiling = -(values[0] + values[1])
            if floor > ceiling:
                return []
            iLeft = nvalues
            iRight = -1
            for i in range(nvalues):
                if values[i] >= floor:
                    iLeft = i
                    break
            for i in range(nvalues - 1, -1, -1):
                if values[i] <= ceiling:
                    iRight = i
                    break
            if iLeft == 0 and iRight == nvalues - 1:
                break
            values = values[iLeft:iRight + 1]
            nvalues = len(values)
        if nvalues < 3:
            return []

        result = []
        # special case for (0,0,0), otherwise v1 must be negative
        if 0 in instances and instances[0] == 3:
            result.append([0, 0, 0])
        for i in range(nvalues - 2):
            v1 = values[i]
            if v1 >= 0:
                break
            if i > 0 and v1 == values[i - 1]:
                continue
            for j in range(i + 1, nvalues - 1):
                v2 = values[j]
                if j > i + 1 and v2 == values[j - 1]:
                    continue
                v3 = -(v1 + v2)
                if v3 < v2:
                    break
                if v3 in instances:
                    if v3 > v2 or instances[v3] > 1:
                        result.append([v1, v2, v3])
        return result


s = Solution()
s.threeSum([-1, 0, 1, 2, -1, -4])
