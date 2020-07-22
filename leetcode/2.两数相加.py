#
# @lc app=leetcode.cn id=2 lang=python3
#
# [2] 两数相加
#

# @lc code=start
# Definition for singly-linked list.


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        a = []
        b = []
        while l1:
            a.append(str(l1.val))
            l1 = l1.next
        while l2:
            b.append(str(l2.val))
            l2 = l2.next
        a.reverse()
        b.reverse()
        a = ''.join(a)
        b = ''.join(b)
        c = str(int(a)+int(b))
        c = [int(x) for x in c]
        c.reverse()
        for idx, item in enumerate(c):
            if idx == 0:
                res = p = ListNode(item)
            else:
                p.next = ListNode(item)
                p = p.next
        return res
# @lc code=end
