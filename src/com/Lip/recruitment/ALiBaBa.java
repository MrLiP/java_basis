package com.Lip.recruitment;

import java.util.*;

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
    public static int getRes(int[] arr){
        if (arr.length==2) return 0;
        Map<Integer,Integer> odd = new HashMap<>();
        Map<Integer,Integer> even = new HashMap<>();
        for (int i = 0; i <arr.length ; i++) {
            if (i % 2==0){
                odd.put(arr[i], odd.getOrDefault(arr[i],0)+1);
            }else{
                even.put(arr[i], even.getOrDefault(arr[i],0)+1);
            }
        }
        int r1 = 1, r2 = 1;
        for (int val : odd.values()){
            r1 = Math.max(r1, val);
        }
        for (int val:even.values()){
            r2 = Math.max(r2, val);
        }
        return arr.length - r1 - r2;
    }

    public static int solution_3(int n, int m, int x1, int y1, int x2, int y2, char[][] chs) {

        if (y2 + 1 < chs.length) {
            check(x1, y1, x2, y2 + 1, chs, 0);
        } else if (y2 - 1 > 0) {
            check(x1, y1, x2, y2 - 1, chs, 0);
        } else if (x2 - 1 > 0) {
            check(x1, y1, x2 - 1, y2 - 1, chs, 0);
        } else if (x2 + 1 < chs[0].length) {
            check(x1, y1, x2 + 1, y2, chs, 0);
        } else if (x2 - 1 > 0 && y2 - 1 > 0) {
            check(x1, y1, x2 - 1, y2 - 1, chs, 0);
        } else if (x2 + 1 < chs[0].length && y2 - 1 > 0) {
            check(x1, y1, x2 + 1, y2 - 1, chs, 0);
        } else if (x2 - 1 > 0 && y2 + 1 < chs.length) {
            check(x1, y1, x2 - 1, y2 + 1, chs, 0);
        } else if (x2 + 1 < chs[0].length && y2 + 1 < chs.length) {
            check(x1, y1, x2 + 1, y2 + 1, chs, 0);
        }

        return ans;
    }

    static int ans;
    public static void check(int x1, int y1, int x2, int y2, char[][] chs, int temp) {
        if (ans > -1 && temp > ans) return;
        if (x1 > chs[0].length || x1 < 0 || y1 > chs.length || y1 < 0 || chs[x1][y1] == '#' || chs[x2][y2] == '#') return;

        if (x1 == x2 && y1 == y2) {
            if (ans != -1) ans = Math.min(temp, ans);
            else ans = temp;
        }
        check(x1 + 1, y1, x2, y2, chs, temp + 1);
        check(x1 - 1, y1, x2, y2, chs, temp + 1);
        check(x1, y1 + 1, x2, y2, chs, temp + 1);
        check(x1, y1 - 1, x2, y2, chs, temp + 1);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int T = sc.nextInt();
        int[] result = new int[T];

        for (int i = 0; i < T; i++) {
            int n = sc.nextInt();
            int m = sc.nextInt();
            int x1 = sc.nextInt(), y1 = sc.nextInt(), x2 = sc.nextInt(), y2 = sc.nextInt();
            char[][] chs = new char[n][m];
            for (int j = 0; j < n; j++) {
                chs[j] = sc.next().toCharArray();
            }
            result[i] = solution_3(n, m, x1 - 1, y1 - 1, x2 - 1, y2 - 1, chs);
        }

        for (int res : result) {
            System.out.println(res);
        }
    }
}
