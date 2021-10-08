package com.Lip.recruitment;

import java.util.Scanner;

public class HuaWei {


    public static void main(String[] args) {
        // please define the JAVA input here. For example: Scanner s = new Scanner(System.in);
        // please finish the function body here.
        // please define the JAVA output here. For example: System.out.println(s.nextInt());
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();

        int[] nums = new int[t];

        for (int i = 0; i < t; i++) {
            nums[i] = sc.nextInt();
        }

        System.out.println(solution_3(nums, t));

        System.out.println(solution_1("0002:0000:0000:0000:0000:0100:0000:0000"));
    }

    private static String solution_1(String str) {
        if (str.length() != 39) {
            return "error";
        }
        StringBuilder sb = new StringBuilder();
        char[] chs = str.toCharArray();

        int n = 0;
        boolean flag = true;
        boolean fir = false;
        for (int i = 0; i < chs.length; i++) {
            if (i % 5 == 0) fir = true;
            else fir = false;

            //if (fir && chs[i] != '0') {
            //    while (chs[i] != ':') {
            //        sb.append(chs[i++]);
            //    }
            //    sb.append(chs[i++]);
            //}
            if (fir && chs[i] == '0') {
                if (flag) {
                    int k = 0;
                    while (i < chs.length - 1 && (chs[i] == '0' || chs[i] == ':')) {
                        k++;
                        i++;
                    }
                    if (k >= 10) {
                        flag = false;
                        i -= k;
                        i += (k / 5 * 5) - 1;
                    } else if (k >= 5) {
                        sb.append("0");
                        i -= k;
                        i += (k / 5 * 5) - 1;
                    }
                } else {
                    while (i < chs.length - 1 && chs[i] == '0') {
                        i++;
                    }
                    if (chs[i] == ':') sb.append("0");
                }
            }

            sb.append(chs[i]);
        }

        return sb.toString();
    }

    private static long solution_3(int[] nums, int t) {
        long ans = 0;
        int[][][] dp = new int[t][t][2];
        // 0 store max , 1 store min;

        for (int i = 0; i < t; i++) {
            dp[i][i][0] = nums[i];
            dp[i][i][1] = nums[i];
        }

        for (int i = 0; i < t - 1; i++) {
            for (int j = i + 1; j < t; j++) {
                dp[i][j][0] = Math.max(dp[i][j - 1][0], nums[j]);
                dp[i][j][1] = Math.min(dp[i][j - 1][1], nums[j]);

                ans += (dp[i][j][0] - dp[i][j][1]) * (j - i + 1);
                ans %= 1000000007;
            }
        }

        return ans;
    }

    private static void swap(int a, int b, int[] nums) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }
}


