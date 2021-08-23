package com.Lip.recruitment;

import java.util.Arrays;

public class WangYi {

    public static void main(String[] args) {

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
