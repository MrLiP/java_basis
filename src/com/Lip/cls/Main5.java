package com.Lip.cls;


import java.lang.reflect.Array;
import java.util.*;
import com.Lip.TreeNode;

public class Main5{
    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.right.left = new TreeNode(4);
        root.right.right = new TreeNode(5);

//        System.out.println(minM(17, 10));
        System.out.println(cyclicShiftTree(root, 1));
    }

    public static TreeNode cyclicShiftTree (TreeNode root, int k) {
        // write code here
        Deque<TreeNode> deque1 = new LinkedList<TreeNode>();
        Deque<TreeNode> deque2 = new LinkedList<TreeNode>();
        Deque<TreeNode> deque3;
        deque1.offer(root);

        boolean flag = true;
        while (!deque1.isEmpty() && flag) {
            int size = deque1.size();
            flag = false;
            deque3 = new LinkedList<TreeNode>(deque1);

            for (int i = 0; i < size; i++) {
                TreeNode node = deque1.poll();

                if (node.left != null) {
                    deque2.offer(node.left);
                    flag = true;
                } else {
                    deque2.offer(new TreeNode(-1));
                }
                if (node.right != null) {
                    deque2.offer(node.right);
                    flag = true;
                } else {
                    deque2.offer(new TreeNode(-1));
                }

                for (int j = 0; j < k; j++) {
                    deque2.offer(deque2.poll());
                }
            }
            deque1 = new LinkedList<TreeNode>(deque2);
            System.out.println(deque1.toString());

            for (int i = 0; i < size; i++) {
                TreeNode node = deque3.poll();

                TreeNode left = deque2.poll();
                TreeNode right = deque2.poll();

                if (left.val != -1) {
                    node.left = left;
                } else node.left = null;
                if (right.val != -1) {
                    node.right = right;
                } else node.right = null;
            }
        }

        return root;
    }

    public static long minM (int n, int k) {
        // write code here
        long ans = 0;

        int digit = 1, high = n / k, cur = n % k, low = 0;

        while (high != 0 || cur != 0) {
            ans += high * digit;
            if (cur == 1) {
                ans += low + 1;
            } else if (cur != 0) {
                ans += digit;
            }

            low += cur * digit;
            cur = high % k;
            high /= k;
            digit *= k;
        }

        return ans;
    }
}


