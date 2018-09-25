class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        dicts = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        for i in range(1, len(s)):
            if dicts[s[i - 1]] < dicts[s[i]]:
                result -= dicts[s[i - 1]]
            else:
                result += dicts[s[i - 1]]
        return result + dicts[s[-1]]
