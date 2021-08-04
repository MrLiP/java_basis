package com.Lip.recruitment;

import java.util.ArrayList;
import java.util.List;

public class DuoYi {
    public static void main(String[] args) {
        System.out.println("hello");

    }

    // 给定一个集合S(没有重复元素), 输出它所有的子集 输入 1 2 3 输出 1, 2, 12, 3, 13, 23, 123

    /**
     * public List<List<Integer>> subsets(int[] nums) {
     *         List<List<Integer>> res = new ArrayList<>();
     *         List<Integer> stack = new ArrayList<>();
     *
     *         dfs(nums, stack, res, 0);
     *         return res;
     *     }
     *
     *     public void dfs(int[] nums, List<Integer> stack, List<List<Integer>> res, int index) {
     *         if (index == nums.length) {
     *             res.add(new ArrayList<>(stack));
     *             return;
     *         }
     *
     *         stack.add(nums[index]);
     *         dfs(nums, stack, res, index+1);
     *         stack.remove(stack.size() - 1);
     *         dfs(nums, stack, res, index+1);
     *     }
     * @param nums
     * @return
     */

}
