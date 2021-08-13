package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.lang.reflect.Array;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Scanner;

public class BeiKe {
    public static void main(String[] args) {
        int n = 2, m = 5;

        System.out.println(Arrays.toString(solution_1(n, m)));
//        System.out.println(solution_2(str, n));
    }

    private static long[] solution_1(int n, long m) {
        long[] ans = new long[n];

        long k = m / (n - 1);
        long p = m % (n - 1);

        Arrays.fill(ans, k);
        ans[0] = (k + 1) / 2;
        ans[n - 1] = k / 2;
        if (k % 2 == 0) {
            for (int i = 0; i < p; i++) ans[i]++;
        } else {
            for (int i = 0; i < p; i++) ans[n - 1 - i]++;
        }

        return ans;
    }

    private static String solution_2(String str, int n) {
        int[] word = new int[26];
        char[] chs = str.toCharArray();

        for (Character ch : chs) {
            word[ch - 'a'] += 1;
        }
        int i = 0;
        while (n > 0) {
            if (word[i] != 0) n--;
            i++;
        }

        StringBuilder sb = new StringBuilder();
        for (Character ch : chs) {
            if (ch - 'a' >= i) sb.append(ch);
        }
        return sb.toString();
    }
}
