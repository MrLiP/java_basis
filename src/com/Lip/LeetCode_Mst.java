package com.Lip;

import java.util.HashSet;

class Solution_Mst {

    // 面试题 01.01. 判定字符是否唯一
    // HashMap, int[26] 统计所有的字母
    // 基于位运算的方法，使用一个int类型的变量（下文用mark表示）来代替长度为26的bool数组
    public boolean isUnique(String astr) {
        int[] chs = new int[26];

        for(int i = 0; i < astr.length(); i++) {
            char ch = astr.charAt(i);
            chs[ch - 'a'] += 1;
        }

        for (int i : chs) {
            if (i > 1) return false;
        }
        return true;
    }

    // 面试题 01.02. 判定是否互为字符重排
    // 1.先排序再比较是否相等，s1.toCharArray(), Arrays.sort(), new String(...).equals(new String(...))
    // 2. HashMap 计数
    // 3. 桶计数，即用数组来统计每个字符次数，本质类似方法二
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;

        int[] chs = new int[26];

        for (int i = 0; i < s1.length(); i++) {
            chs[s1.charAt(i) - 'a'] += 1;
            chs[s2.charAt(i) - 'a'] -= 1;
        }

        for (int i : chs) {
            if (i != 0) return false;
        }
        return true;
    }

    // 面试题 01.03. URL化
    // 倒序计算，从后往前为数组赋值
    // StringBuilder
    // replace(), substring()
    public String replaceSpaces(String S, int length) {
        char[] chars = S.toCharArray();
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < length; i++) {
            char ch = S.charAt(i);
            if (ch == ' ') {
                str.append("%20");
            } else str.append(ch);
        }
        return str.toString();
    }

    // 面试题 01.04. 回文排列
    // set 去重，偶数情况直接去除，最后判断 set 的 size()
    // 位运算，用两个 long 判断 128 位的字符，一个 long 位数为 64位 (8字节)，int 为 32位
    public boolean canPermutePalindrome(String s) {
        int[] chs = new int[128];

        for (char ch : s.toCharArray()) {
            chs[ch] += 1;
        }

        boolean flag = false;
        for (int i : chs) {
            if (i % 2 == 1) {
                if (!flag) flag = true;
                else return false;
            }
        }
        return true;
    }

    public boolean canPermutePalindrome_opt(String s) {
        int[] map = new int[128];
        int count = 0;
        for (char ch : s.toCharArray()) {
            //如果当前位置的字符数量是奇数，再加上当前字符
            //正好是偶数，表示消掉一个，我们就把count减一，
            //否则count就加一
            if ((map[ch]++ & 1) == 1) {
                count--;
            } else {
                count++;
            }
        }
        return count <= 1;
    }

    // 面试题 01.05. 一次编辑
    // 动态规划，双指针模拟
    public boolean oneEditAway(String first, String second) {
        if (Math.abs(first.length() - second.length()) > 1) return false;

        int[][] dp = new int[first.length() + 1][second.length() + 1];

        for (int i = 1; i < first.length(); i++) dp[0][i] = i - 1;
        for (int j = 1; j < second.length(); j++) dp[j][0] = j - 1;

        for (int i = 1; i <= first.length(); i++) {
            for (int j = 1; j <= second.length(); j++) {
                if (first.charAt(i - 1) == second.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = dp[i - 1][j - 1] + 1;
                dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
                dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
            }
        }

        return dp[first.length()][second.length()] <= 1;
    }

    // 面试题 01.06. 字符串压缩
    // StringBuilder, 比较前后是否一致
    public String compressString(String S) {
        StringBuilder str = new StringBuilder();

        for (int i = 0; i < S.length(); i++) {
            int n = 1;
            while (i + 1 < S.length() && S.charAt(i) == S.charAt(i + 1)) {
                n += 1;
                i += 1;
            }
            str.append(S.charAt(i));
            str.append(n);
        }

        return S.length() <= str.length() ? S : str.toString();
    }

    // 面试题 01.07. 旋转矩阵
    // 原地旋转，for (int i = 0; i < n/2; i++)，for (int j = 0; j < (n+1)/2; j++)
    // matrix[i][j] << matrix[n - j - 1][i] << matrix[n - i - 1][n - j - 1] << matrix[j][n - i - 1]
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int mid = n / 2;

        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < (n+1)/2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }

    // 面试题 01.08. 零矩阵
    // HashSet, 遍历两次，第一次保存 0 的行列值，第二次将对应行列置 0
    // HashSet >> boolean[]，优化思路
    // 优化思路，利用矩阵的第一行和第一列来代替之前的标记数组
    public void setZeroes(int[][] matrix) {
        HashSet<Integer> row = new HashSet<>();
        HashSet<Integer> col = new HashSet<>();

        int m = matrix.length, n = matrix[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row.add(i);
                    col.add(j);
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row.contains(i) || col.contains(j)) matrix[i][j] = 0;
            }
        }
    }

    // 面试题 01.09. 字符串轮转
    // 先判断长度是否相同，不相同返回false，其次拼接两个s2，则如果是由s1旋转而成，则拼接后的s一定包含s1
    // String s = s1 + s2;  return s.contains(s1);
    public boolean isFlipedString(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        if (s1.equals(s2) || s1.length() == 0) return true;

        for (int i = 0; i < s1.length(); i++) {
            int j = 0;
            if (s2.charAt(i) == s1.charAt(j)) {
                while (j < s2.length() && s2.charAt((i + j) % s2.length()) == s1.charAt(j)) {
                    j++;
                }
                if (j == s2.length()) return true;
            }
        }

        return false;
    }

    // 面试题 02.01. 移除重复节点

}

public class LeetCode_Mst {
    public static void main(String[] args) {
//        int[] arr = {1,2,3};
//        List list = Arrays.asList(1,2,3);
        Solution_Mst solution = new Solution_Mst();
        int[] nums1 = new int[]{23,2,6,4,7};
        int[] nums2 = new int[]{2};
        System.out.println(solution.isFlipedString("waterbottle", "erbottlewat"));
    }
}
