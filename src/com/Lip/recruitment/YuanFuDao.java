package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class YuanFuDao {
    public List<List<Integer>> zhiTree(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) queue.offer(root);

        while (!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                assert node != null;

                if (res.size() % 2 == 0) tmp.addLast(node.val);
                else tmp.addFirst(node.val);

                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }

    public static void main(String[] args) {
        System.out.println("hello");
    }
}
