package com.Lip;

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

}

public class LeetCode_Mst {
    public static void main(String[] args) {
//        int[] arr = {1,2,3};
//        List list = Arrays.asList(1,2,3);
        Solution_Mst solution = new Solution_Mst();
        int[] nums1 = new int[]{23,2,6,4,7};
        int[] nums2 = new int[]{2};
        System.out.println(solution.isUnique("abcdef"));
    }
}
