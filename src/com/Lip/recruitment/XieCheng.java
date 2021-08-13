package com.Lip.recruitment;

import java.util.HashMap;
import java.util.Scanner;


public class XieCheng {
    static HashMap<Character, Integer[]> map = new HashMap<Character, Integer[]>(){{
        put('0', new Integer[]{1,1,1,0,1,1,1});
        put('1', new Integer[]{0,0,1,0,0,1,0});
        put('2', new Integer[]{1,0,1,1,1,0,1});
        put('3', new Integer[]{1,0,1,1,0,1,1});
        put('4', new Integer[]{0,1,1,1,0,1,0});
        put('5', new Integer[]{1,1,0,1,0,1,1});
        put('6', new Integer[]{1,1,0,1,1,1,1});
        put('7', new Integer[]{1,0,1,0,0,1,0});
        put('8', new Integer[]{1,1,1,1,1,1,1});
        put('9', new Integer[]{1,1,1,1,0,1,1});
        put(' ', new Integer[]{0,0,0,0,0,0,0});
    }};

    public static int[] solution_2(String str) {
        int length = str.length();
        int[] res = new int[length];

        for (int l = 1; l <= length; l++) {
            res[l - 1] = change(str, l);
        }

        return res;
    }

    private static int change(String str, int l) {
        int ans = 0;

        char[] pre = new char[l], cur;
        for (int i = 0; i < str.length() + 1 - l; i++) {
            cur = str.substring(i, i + l).toCharArray();
            if (i == 0) {
                for (Character c : cur) {
                    int temp = 0;
                    for (int k : map.get(c)) {
                        temp += k;
                    }
                    ans += temp;
                }
            } else {
                for (int p = 0; p < l; p++) {
                    int temp = 0;
                    for (int q = 0; q < 6; q++) {
                        temp += Math.abs(map.get(pre[p])[q] - map.get(cur[p])[q]);
                    }
                    ans += temp;
                }
            }
            pre = cur;
        }
        return ans;
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String str = sc.nextLine();

        int[] res = solution_2(str);
        for (int i = 0; i < res.length; i++) {
            System.out.print(res[i] + " ");
        }

    }
}
