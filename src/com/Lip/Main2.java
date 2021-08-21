package com.Lip;

import java.util.*;

public class Main2 {

    static int ans = 0;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[][] nums = new int[n][n];
        int zeros = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                String temp = sc.next();
                if (temp.equals(".")) {
                    nums[i][j] = 0;
                    zeros++;
                }
                else nums[i][j] = 1;
            }
        }

        dfs(nums, n, 0, 0, zeros);

        System.out.println(ans);
    }

    private static void dfs(int[][] nums, int n, int row, int col, int zeros) {
        if (nums[row][col] == 1) return;

        nums[row][col] = 1;
        zeros--;

        if (row == n - 1 && zeros == 0) {
            ans++;
            return;
        }

        if (row - 1 >= 0) dfs(nums, n, row - 1, col, zeros);
        if (row + 1 < n) dfs(nums, n, row + 1, col, zeros);
        if (col - 1 >= 0) dfs(nums, n, row, col - 1, zeros);
        if (col + 1 < n) dfs(nums, n, row, col + 1, zeros);
        nums[row][col] = 0;
    }

    private static boolean check(int[][] nums) {
        int n = nums.length;
        for (int[] num : nums) {
            for (int j = 0; j < n; j++) {
                if (num[j] == 0) return false;
            }
        }
        return true;
    }
}
