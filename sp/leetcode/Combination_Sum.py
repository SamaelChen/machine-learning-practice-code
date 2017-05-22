"""
Given a set of candidate numbers (C)
(without duplicates) and a target number (T),
find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [2, 3, 6, 7] and target 7,
A solution set is:
[
  [7],
  [2, 2, 3]
]
"""


class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers

    def combinationSum(self, candidates, target):
        ans = []
        candidates.sort()
        for ii, elem in enumerate(candidates):
            if target > elem:
                subSet = self.combinationSum(candidates[ii:], target - elem)
                # print(subSet)
                # need to update the candidates list to avoid dublicates
                if subSet:
                    ans += [[elem] + lis for lis in subSet]
                    # print(ans)
                    # print('=======')
            elif target == elem:
                ans += [[elem]]
            else:
                break
        return ans


s = Solution()
s.combinationSum([1, 2, 3, 6, 7], 6)
candidates = [2, 3, 6, 7]
for i, elem in enumerate(candidates):
    print(i, elem)
candidates[1:]
elem

ans = []
[[1] + [[2], 3]]
