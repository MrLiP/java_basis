package com.Lip.recruitment;

import java.util.Arrays;
import java.util.Scanner;

public class ALiBaBa {

    public static int solution_1(int[] nums) {
        int ans = 0, left = 0, sum = 0, max = 0;
        for (int num : nums) {
            sum += num;
        }

        for (int i = 0; i < nums.length; i++) {
            left += nums[i];
            if (left * (sum - left) > max) {
                max = left * (sum - left);
                ans = i;
            }
        }
        return ans + 1;
    }

//    public static int[] solution_2(int n, int l, int r) {
//        int[] ans = new int[r - l + 1];
//
//        for (int j = 1; j < n; j++) {
//            int k = n % j;
//            while(k > 0 && (n - k) < r) {
//                ans[n - k - l + 1] += 1;
//                k--;
//            }
//        }
//
//        return ans;
//    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String[] str = sc.nextLine().split(" ");
        long n = Long.parseLong(str[0]);
        long l = Long.parseLong(str[1]);
        long r = Long.parseLong(str[2]);

        long[] ans = new long[(int) (r - l + 1)];

        for (long j = 1; j < n; j++) {
            long k = n % j;
            while(k > 0 && (n - k) < r) {
                ans[(int) (n - k - l + 1)] += 1;
                k--;
            }
        }

        for (long i : ans) {
            System.out.print(i + " ");
        }
    }
}
