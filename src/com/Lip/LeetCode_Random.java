package com.Lip;

import java.util.*;

public class LeetCode_Random {
    public static void main(String[] args) {
        System.out.println();
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;

        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
