package com.Lip.algorithm;

/**
 * KMP 算法：
 * KMP 要解决的问题就是在字符串（也叫主串）中的模式（pattern）定位问题。说简单点就是我们平时常说的关键字搜索
 * 利用已经部分匹配这个有效信息，保持i指针不回溯，通过修改j指针，让模式串尽量地移动到有效的位置。
 * 整个KMP的重点就在于当某一个字符与主串不匹配时，我们应该知道j指针要移动到哪？
 *
 * 当匹配失败时，j要移动的下一个位置k。存在着这样的性质：最前面的k个字符和j之前的最后k个字符是一样的。
 * 用数学公式表示：P[0 ~ k-1] == P[j-k ~ j-1]
 *
 * 我们要计算每一个位置j对应的k，所以用一个数组next来保存，next[j] = k，表示当T[i] != P[j]时，j指针的下一个位置。
 */
public class KMP {
    public static void main(String[] args) {
        System.out.println(kmp("absabcabc", "sabc"));
    }

    public static int kmp(String ts, String ps) {
        char[] t = ts.toCharArray();
        char[] p = ps.toCharArray();
        int i = 0; // 主串的位置
        int j = 0; // 模式串的位置
        int[] next = getNext(ps);

        while (i < t.length && j < p.length) {
            if (j == -1 || t[i] == p[j]) { // 当j为-1时，要移动的是i，当然j也要归0
                i++;
                j++;
            } else {
                // i不需要回溯了
                // i = i - j + 1;
                j = next[j]; // j回到指定位置
            }
        }

        if (j == p.length) {
            return i - j;
        } else {
            return -1;
        }
    }

    public static int[] getNext(String ps) {
        char[] p = ps.toCharArray();
        int[] next = new int[p.length];
        next[0] = -1;
        int j = 0;
        int k = -1;

        while (j < p.length-1) {
            if (k == -1 || p[j] == p[k]) {
                j++; k++;
                if (p[j] == p[k]) //当两个字符相同时，就跳过
                    next[j] = next[k];
            }
            else k = next[k];
        }

        return next;
    }
}
