package com.Lip.recruitment;

import java.util.Arrays;

public class WangYi {

    public static void main(String[] args) {
        int s = 39;
        // if (s == 0) System.out.println(-1);

        solution_2_incr(s);
        solution_2_decr(s);

        System.out.println(Math.min(count1, count2));
    }

    static int count1 = 0;
    static int count2 = 0;

    private static void solution_2_incr(double s) {
        int a = (int)(Math.log(s)/Math.log(2));
        count1++;

        if (Math.pow(2, a) == s) {
            return;
        }

        solution_2_incr(s - Math.pow(2, a));
    }

    private static void solution_2_decr(double s) {
        int a = (int)(Math.log(s)/Math.log(2));
        count2++;

        if (Math.pow(2, a) == s) {
            return;
        }

        solution_2_decr(Math.pow(2, a + 1) - s);
    }

    // "azbA5#1@c"
    public static long convertMagicalString (String magicalString) {
        StringBuilder word = new StringBuilder();
        StringBuilder num = new StringBuilder();

        for (char ch : magicalString.toCharArray()) {
            if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
                if ((ch >= '0' && ch <= '9')) num.append(ch);
                else {
                    if (ch >= 'A' && ch <= 'Z') word.append((char)(ch + 32));
                    else word.append(ch);
                }
            }
        }
        char[] nums = num.toString().toCharArray();
        char[] words = word.toString().toCharArray();
        Arrays.sort(nums);
        Arrays.sort(words);

        StringBuilder ans = new StringBuilder();

        for (int i = 0; i < words.length; i++) {
            while(i + 1 < words.length && words[i + 1] - words[i] == 1) {
                i++;
            }
            ans.append(words[i] - 'a' + 1);
        }

        for (char ch : nums) ans.append(ch);


        System.out.println(Arrays.toString(words));
        System.out.println(Arrays.toString(nums));
        System.out.println(ans.toString());

        return Long.parseLong(ans.toString());
    }
}
