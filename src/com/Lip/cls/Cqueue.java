package com.Lip.cls;

import java.util.LinkedList;

class Aqueue {
    LinkedList<Integer> A, B;
    public Aqueue() {
        A = new LinkedList<>();
        B = new LinkedList<>();
    }

    public void appendTail(int value) {
        A.addLast(value);
    }

    public int deleteHead() {
        if (!B.isEmpty()) return B.removeLast();
        if (A.isEmpty()) return -1;
        while (!A.isEmpty()) {
            B.addLast(A.removeLast());
        }
        return B.removeLast();
    }
}

public class Cqueue {
    public static void main(String[] args) {
        Aqueue queue = new Aqueue();
        queue.appendTail(5);
        System.out.println(queue.deleteHead());
    }
}
