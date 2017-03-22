"""
You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        newlist = [0] * (len(l1) + 1)
        for i in range(len(l1)):
            newlist[i] = newlist[i] + (l1[i] + l2[i]) % 10
            if l1[i] + l2[i] >= 10:
                newlist[i + 1] += 1
        return newlist[0:len(l1)]


s = Solution()
s.addTwoNumbers([2, 3, 4], [6, 7, 8])
