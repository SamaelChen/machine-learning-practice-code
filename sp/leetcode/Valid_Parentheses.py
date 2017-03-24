"""
Given a string containing just the characters '(', ')', '{', '}', '[' and ']',
determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
"""


class Solution(object):

    def isValid(self, s):
        dic = {')': '(', '}': '{', ']': '['}
        st = []
        for e in s:
            if st and (e in dic and st[-1] == dic[e]):
                st.pop()
            else:
                st.append(e)
        return not st


s = Solution()
s.isValid('([])[]')
s.isValid('][()][')
s.isValid('[(]))')
