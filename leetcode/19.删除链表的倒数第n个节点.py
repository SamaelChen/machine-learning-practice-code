#
# @lc app=leetcode.cn id=19 lang=python3
#
# [19] 删除链表的倒数第N个节点
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dum = ListNode(0)
        dum.next = head
        cur = head
        pre = dum
        for _ in range(n):
            cur = cur.next
        while cur:
            cur = cur.next
            pre = pre.next
        pre.next = pre.next.next
        return dum.next
# @lc code=end

