package com.Lip.recruitment;

import java.util.HashMap;
import java.util.Scanner;


public class XieCheng {

    // solution 1
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        StringBuilder cur = new StringBuilder("\\");
        int[] nums = new int[n + 1];
        int index = 0;
        int num = 1;
        nums[index++] = num;

        for (int i = 0; i < n ;i++) {
            String a = sc.next();

            if (a.equals("pwd")) {
                System.out.println(cur.toString());
            } else if (a.equals("cd")) {
                String b = sc.next();

                if (b.equals("..")) {
                    int len = cur.length();
                    if(len != 1) {
                        num = nums[index - 2];
                        cur.delete(nums[index - 2], len);
                        index--;
                        /*
                        while (cur.charAt(len - 1) != '\\') {
                            cur.deleteCharAt(len - 1);
                            len--;
                        }
                        if (len == 1) break;
                        cur.deleteCharAt(len - 1);
                         */
                    }
                } else {
                    if (cur.length() != 1) {
                        cur.append("\\");
                        num++;
                    }
                    cur.append(b);
                    num += b.length();
                    nums[index] = num;
                    index++;
                }
            }
        }
    }
}
