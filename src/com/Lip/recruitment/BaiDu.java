package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.util.*;

public class BaiDu {

    /*
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int k = sc.nextInt();
        int[][] arr = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                arr[i][j] = sc.nextInt();
            }
        }

        int[][] res = new int[n * k][n * k];
        res = solution_1(arr, n, k);

        for (int i = 0; i < n * k; i++) {
            for (int j = 0; j < n * k; j++) {
                System.out.print(res[i][j] + " ");
            }
            System.out.println("");
        }
    }


     */
    private static int[][] solution_1(int[][] arr, int n, int k) {
        int[][] res = new int[n * k][n * k];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int p = 0; p < k; p++) {
                    for (int q = 0; q < k; q++) {
                        res[i * k + p][j * k + q] = arr[i][j];
                    }
                }
            }
        }

        return res;
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        long[] nums = new long[t];
        for (int i = 0; i < t; i++) {
            nums[i] = sc.nextLong();
        }

        for (int i = 0; i < t - 1; i++) {
            // System.out.println(String.valueOf(solution_2(nums[i])));
            System.out.println(solution_2(nums[i]));
        }
        if (t != 0) System.out.print(solution_2(nums[t - 1]));
    }

    private static long solution_2(long num) {
        char[] chs = String.valueOf(num).toCharArray();
        int len = chs.length;

        long top = 3;
        long bottom = 1;
        for (int i = 1; i < len; i++) {
            top *= 10;
            top += 3;
            bottom *= 10;
            bottom += 1;
        }

        if (num >= top) return top;
        if (num < bottom) return top / 10;

        boolean flag = false;
        boolean cur = true;
        for (int i = 0; i < len; i++) {
            if (!cur) chs[i] = '3';
            else {
                if (chs[i] >= '1' && chs[i] <= '3') {
                    flag = true;
                    continue;
                }
                else if (chs[i] > '3') {
                    chs[i] = '3';
                    flag = false;
                }
                else {
                    if (flag && cur) {
                        chs[i - 1]--;
                        flag = false;

                    }
                    chs[i] = '3';
                    cur = false;
                }
            }
        }
        return Long.parseLong(String.valueOf(chs));
    }
}


