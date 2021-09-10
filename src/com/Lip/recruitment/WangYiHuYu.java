package com.Lip.recruitment;

import java.util.*;

public class WangYiHuYu {

    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String str = sc.next();
//        int k = sc.nextInt();

//        System.out.println(solution_2(str, k));
        System.out.println(canBePalindromicString("abcdefgh"));
    }

    public static int canBePalindromicString (String str1) {
        if (str1.length() <= 2) return 1;

        //boolean flag = false;
        int len = str1.length();
        if (len > 500000) return 1;

        for (int i = 0; i < len - 1; i++) {
            String temp = str1.substring(0, i) + str1.substring(i + 2, len);
            if (check(temp)) return 1;
        }

        return 0;
    }

    private static boolean check(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) return false;
        }
        return true;
    }

    private static int solution_2(String str, int k) {
        int ans = 0, left = 0, right = 0;
        char[] chs = str.toCharArray();
        int len = chs.length;

        while (right < len) {
            if (chs[right] == '0') {
                if (k == 0) {
                    while (chs[left] == '1') left += 1;
                    left += 1;

                } else {
                    k -= 1;
                }
            }
            ans = Math.max(ans, ++right - left);

        }

        return ans;
    }

    public int find_kth(int[] arr1, int[] arr2, int k) {
        int len1 = arr1.length, len2 = arr2.length;
        if (len1 == 0) return arr2[k];
        if (len2 == 0) return arr1[k];
        if (k == 1) return Math.min(arr1[0], arr2[0]);
        if (k == len1 + len2) return Math.max(arr1[len1 - 1], arr2[len2 - 1]);

        int[] a1 = len1 < len2 ? arr1 : arr2; // s
        int[] a2 = (len1 >= len2) ? arr1 : arr2;

        len1 = a1.length;
        len2 = a2.length; // l

        if (k > len2) {
            if (a1[k - len2 - 1] >= a2[len2 - 1]) return a1[k - len2 - 1];
            if (a2[k - len1 - 1] >= a1[len1 - 1]) return a2[k - len1 - 1];
            return getK(a1, k - len2, len1 - 1, a2, k - len1, len2 - 1);
        }
        if (k <= len1) {
            return getK(a1, 0, k - 1, a2, 0, k - 1);
        }
        if (a2[k - len1 - 1] >= a1[len1 - 1]) return a2[k - len1 - 1];

        return getK(a1, 0, len1 - 1, a2, k - len1, k - 1);
    }

    private int getK(int[] a1, int l1, int r1, int[] a2, int l2, int r2) {
        int p1 = 0, p2 = 0, p3 = 0;
        while (l1 < r1) {
            p1 = (l1 + r1) / 2;
            p2 = (l2 + r2) / 2;
            p3 = ((r1 - l1 + 1) & 1) ^ 1;
            if (a1[p1] > a2[p2]) {
                r1 = p1;
                l2 = p2 + p3;
            } else if (a1[p1] < a2[p2]) {
                l1 = p1 + p3;
                r2 = p2;
            } else {
                return a1[p1];
            }
        }
        return Math.min(a1[l1], a2[l2]);
    }
}



