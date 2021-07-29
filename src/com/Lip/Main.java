package com.Lip;

import java.util.Scanner;


public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        ListNode head = new ListNode(sc.nextInt());
        ListNode cur = head;
        for(int i = 1; i < n; i++){
            cur.next = new ListNode(sc.nextInt());
            cur = cur.next;
        }

        ListNode dummy = new ListNode(0);
        dummy.next = head;
        head = dummy;
        for (int i = 1; i < m; i++) {
            head = head.next;
        }
        head.next = head.next.next;

        for (int i = 0; i < n - 1; i++) {
            dummy = dummy.next;
            System.out.print(dummy.val);
            System.out.print(" ");
        }
    }
}
