package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class ZuiYou {

    public static int solution_1(int[] a, int n) {
        int i;
        int[] visit = new int[n];

        for (i = 0; i < n; i++) {
            if (a[i] > 0 && a[i] <= n) visit[a[i] - 1] = 1;
        }
        for (i = 0; i < n; i++) {
            if (visit[i] == 0) break;
        }

        return i + 1;
    }

    public int firstMissingPositive(int[] nums) {
        for(int i = 0; i < nums.length; i++){
            if(nums[i] <= 0) nums[i] = nums.length + 1;
        }

        for(int i = 0; i < nums.length; i++){
            if(Math.abs(nums[i]) > nums.length) continue;
            else{
                if(nums[Math.abs(nums[i])-1] > 0) nums[Math.abs(nums[i])-1] *= -1;
            }
        }

        for(int i = 0; i < nums.length; i++){
            if(nums[i] > 0){
                return i + 1;
            }
        }

        return nums.length + 1;
    }

    public static int solution_2(int[] stu, int n, int[] darr, int[] uarr) {
        int[] ans = new int[n];
        for (int i = 1; i < n; i++) {
            ans[0] += i * stu[i] * uarr[i];
        }
        if (n == 1) return ans[0];
        for (int i = 0; i < n; i++) {
            darr[i] *= stu[i];
            uarr[i] *= stu[i];
        }
        for (int i = 1; i < n; i++) {
            darr[i] += darr[i - 1];
            uarr[n - i - 1] += uarr[n - i];
        }
        int res = ans[0];
        for (int i = 1; i < n; i++) {
            ans[i] = ans[i - 1];

            ans[i] += darr[i - 1];
            ans[i] -= uarr[i];

            res = Math.min(res, ans[i]);
        }

        return res;
    }

    public static int mianshi_1(int[] a) {
        int ans = Integer.MIN_VALUE, min = a[0];

        for (int i = 1; i < a.length; i++) {
            ans = Math.max(ans, a[i] - min);
            min = Math.min(a[i], min);
        }

        return ans;
    }

    static ArrayList<Integer> tmp = new ArrayList<>();
    public static void mianshi_2(TreeNode root, TreeNode a, TreeNode b) {
        // 两个节点的最长公共路径
        TreeNode together = ggzx(root, a, b);
    }


    public static TreeNode ggzx(TreeNode root, TreeNode a, TreeNode b) {
        if (root == null || root == a || root == b) return root;
        TreeNode left = ggzx(root.left, a, b);
        TreeNode right = ggzx(root.right, a, b);
        return left == null ? right : left;
    }

    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] stu = new int[n], darr = new int[n], uarr = new int[n];
//        for (int i = 0; i < n; i++) {
//            stu[i] = sc.nextInt();
//        }
//        for (int i = 0; i < n; i++) {
//            darr[i] = sc.nextInt();
//            uarr[i] = sc.nextInt();
//        }
        int[] nums1 = new int[]{3,1};
        int[] nums2 = new int[]{2};
        System.out.println(mianshi_1(nums1));
    }
}
