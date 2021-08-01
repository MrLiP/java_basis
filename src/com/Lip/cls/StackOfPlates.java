package com.Lip.cls;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class StackOfPlates {

    private List<Stack<Integer>> list;
    private int cap;

    public StackOfPlates(int cap) {
        this.cap = cap;
        this.list = new ArrayList<Stack<Integer>>();
    }

    public void push(int val) {
        // 如果栈深小于0
        if (cap <= 0) {
            return;
        }
        // 如果当前没有栈，则需要新建栈
        if (list.isEmpty()) {
            list.add(new Stack<Integer>());
        }
        //如果当前最近的栈满，则需要新建栈
        if (list.get(list.size() - 1).size() >= cap) {
            list.add(new Stack<Integer>());
        }
        //再取最近的栈push
        Stack<Integer> stack = list.get(list.size() - 1);
        stack.push(val);
    }

    public int pop() {
        // 如果当前没有栈
        if (list.isEmpty()) {
            return -1;
        }
        // 弹出
        Stack<Integer> stack = list.get(list.size() - 1);
        int val = stack.pop();
        // 如果栈空则要删除栈
        if (stack.isEmpty()) {
            list.remove(list.size() - 1);
        }
        return val;
    }

    public int popAt(int index) {
        // 如果索引不在栈表内
        if(list.size() <= 0 || index > list.size() - 1) {
            return  -1;
        }
        // 弹出
        Stack<Integer> stack = list.get(index);
        Integer pop = stack.pop();
        // 若为空栈则删除
        if(stack.isEmpty()) {
            this.list.remove(index);
        }
        return pop;
    }
}

/**
 * Your StackOfPlates object will be instantiated and called as such:
 * StackOfPlates obj = new StackOfPlates(cap);
 * obj.push(val);
 * int param_2 = obj.pop();
 * int param_3 = obj.popAt(index);
 */
