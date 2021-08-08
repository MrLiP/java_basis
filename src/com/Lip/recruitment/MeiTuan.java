package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.util.*;

public class MeiTuan {

    // TreeSet不支持快速随机遍历，只能通过迭代器进行遍历！
    // int compare(T o1, T o2) o1 比 o2 : 返回负数，o1 比 o2 小，返回 0 相等，返回正数 o1 比 o2 大
    // 只看正数情况，排序中，o1 > o2，
    public static int solution_3(int n, int[] nums) {
        int[] prev = new int[n];
        int ans = 0;
        TreeSet<Integer> tset = new TreeSet<>();
        tset.add(0);

        for (int i = 0; i < n; i++) {
            if (i != 0) {
                if (nums[i] == nums[i - 1]) {
                    prev[i] = prev[i - 1];
                    continue;
                }

                Integer temp = tset.lower(nums[i]);
//                Integer temp1 = tset.ceiling(nums[i]);
                prev[i] = temp == null ? 0 : temp;

//                for (Iterator<Integer> iter = tset.iterator(); iter.hasNext();) {
//                    Integer j = iter.next();
//                    if (nums[i] > j) {
//                        prev[i] = j;
//                        break;
//                    }
//                }
            }
            tset.add(nums[i]);
        }

        for (int i = 0; i < n; i++) {
            ans += ((i + 1) * prev[i]);
        }
        return ans;
    }

    public static TreeNode mirrorTree(TreeNode root) {
        if (root == null) return null;
        TreeNode temp = root.left;

        root.left = mirrorTree(root.right);
        root.right = mirrorTree(temp);
        return root;
    }

    public static void findMirrorNode(TreeNode root, TreeNode node) {
        if (root != null) {
            if (root == node) {
                mirrorTree(node);
                return;
            }
            findMirrorNode(root.left, node);
            findMirrorNode(root.right, node);
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = sc.nextInt();
        }

        System.out.println(solution_3(n, nums));
    }
}
