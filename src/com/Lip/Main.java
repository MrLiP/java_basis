package com.Lip;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.Stack;


public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        String str = sc.next();

        ArrayList<Integer[]> result = solution_3(str, n);

        System.out.println("Test case");
        for (int i = 0; i < result.size() - 1; i++) {
            Integer[] arr = result.get(i);
            System.out.println(arr[0] + " " + arr[1]);
        }
        System.out.print(result.get(result.size() - 1)[0] + " " + result.get(result.size() - 1)[1]);
    }

    private static ArrayList<Integer[]> solution_3(String str, int n) {
        ArrayList<Integer[]> list = new ArrayList<>();

        for (int i = 1; i <= n; i++) {
            String s = str.substring(0, i);
            int temp = check(s);
            if (temp != 0) {
                list.add(new Integer[]{i, i / temp});
            }
        }

        return list;
    }

    private static int check(String s) {
        int n = s.length();

        for (int i = 1; i * 2 <= n; i++) {
            if (n % i == 0) {
                boolean match = true;
                for (int j = i; j < n; j++) {
                    if (s.charAt(j) != s.charAt(j - i)) {
                        match = false;
                        break;
                    }
                }
                if (match) return i;
            }
        }
        return 0;
    }

//    private static int check(String s) {
//        if (s == null || s.length() < 1) return 0;
//        int len = s.length();
//        String str = s;
//        while (len > 1) {
//            str = str.charAt(s.length() - 1) + str.substring(0, s.length() - 1);
//            if (str.equals(s)) return len;
//            len--;
//        }
//        return 0;
//    }

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
