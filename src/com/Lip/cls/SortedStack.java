package com.Lip.cls;

import java.util.Stack;

class SortedStack {
    Stack<Integer> stackIn;

    public SortedStack() {
        stackIn = new Stack<>();
    }

    public void push(int val) {
        stackIn.push(val);
        stackIn.sort((o1, o2) -> (o2 - o1));
    }

    public void pop() {
        if (!stackIn.isEmpty()) stackIn.pop();
    }

    public int peek() {
        if (!stackIn.isEmpty()) return stackIn.peek();
        return -1;
    }

    public boolean isEmpty() {
        return stackIn.isEmpty();
    }
}

/**
 * Your SortedStack object will be instantiated and called as such:
 * SortedStack obj = new SortedStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.isEmpty();
 */
