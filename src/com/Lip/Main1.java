package com.Lip;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Main1 {
    // 两个数组，从小到大
    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String[] str = sc.nextLine().split(",");
//
//        int[] rains = new int[str.length];
//        for(int i = 0; i < rains.length; i++) {
//            if (i == 0) rains[i] = Integer.parseInt(str[i].substring(1, str[i].length()));
//            else if (i == rains.length - 1) {
//                rains[i] = Integer.parseInt(str[i].substring(0, str[i].length()-1));
//            } else {
//                rains[i] = Integer.parseInt(str[i]);
//            }
//        }

//        Scanner sc = new Scanner(System.in);
//        String[] str = sc.nextLine().split(",");
//        String[] lk = str[str.length - 1].split(":");
//        int[] arr = new int[str.length];
//        for (int i = 0; i < arr.length - 1; i++) {
//            arr[i] = Integer.parseInt(str[i]);
//        }
//        arr[arr.length - 1] = Integer.parseInt(lk[0]);
//        int k = Integer.parseInt(lk[1]);
//
////        System.out.println(solution(arr, k));
//        DecimalFormat df = new DecimalFormat("0.00%");
//        String r = df.format(solution(arr, k));
//        System.out.println(r);

//        ListNode head = new ListNode(2);
//        head.next = new ListNode(1);
//        head.next.next = new ListNode(2);
//        head.next.next.next = new ListNode(1);
////        head.next.next.next.next = new ListNode(5);
//
//        ListNode cur = formatList(head);
        System.out.println(minimum(new int[]{1,2,3,4,5}));

    }

    public static long minimum (int[] a) {
        long ans = Long.MAX_VALUE, sum = 0;
        long[] fAdd = new long[a.length + 1];

        for (int i = 0; i < a.length; i++) {
            sum += a[i];
            fAdd[i+1] += (fAdd[i] + a[i]);
        }

        int l = 0, r = 1, temp = a[0];
        while (r < a.length) {
            long cur = sum - 2 * temp;
            if (cur < 0) {
                temp -= a[l];
                l++;
            } else if (cur > 0) {
                temp += a[r];
                r++;
            } else return 0;
            ans = Math.min(ans, Math.abs(cur));
        }

        return ans;

    }

    public static ListNode formatList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return head;

        ListNode dummy = new ListNode(-1), left = head, right = head;
        dummy.next = head;

        int index = 0;
        while (head != null && head.next != null) {
            if (index != 0) {
                ListNode cur = head.next;
                dummy.next = new ListNode(cur.val);
                dummy.next.next = left;
                left = dummy.next;
                head = cur;
                index = 0;
            } else {
                head = head.next;
                right.next = head;
                right = right.next;
                index = 1;
            }
        }

        return dummy.next;
    }
//    private static int[] solution(int[] rains) {
//        int[] ans = new int[rains.length];
//        Map<Integer, Integer> full = new HashMap<>();
//        Deque<Integer> deque = new LinkedList<>();
//
//        for (int i = 0; i < rains.length; i++) {
//            int lake = rains[i];
//            if (lake == 0) {
//                deque.add(i);
//                continue;
//            }
//            if (full.containsKey(lake)) {
//                if (deque.isEmpty()) return new int[0];
//                Integer idx = full.get(lake);
//                boolean flag = false;
//
//                for (int nidx : deque) {
//                    if (nidx > idx) {
//                        ans[nidx] = lake;
//                        deque.remove(nidx);
//                        flag = true;
//                        break;
//                    }
//                }
//                if(!flag) return new int[0];
//            }
//            ans[i] = -1;
//            full.put(lake, i);
//        }
//        while (!deque.isEmpty()) {
//            ans[deque.pop()] = 1;
//        }
//        return ans;
//    }

    // 5,6,8,26,50,48,52,55,10,1,2,1,20,5:3
    private static double solution(int[] arr, int k) {
//        Deque<Integer> deque = new Deque<>();
        double ans = Integer.MIN_VALUE;
        double sum = 0, avg = 0;

        for (int i = 0; i < arr.length; i++) {
            if (i < k) {
                sum += arr[i];
                avg = sum / (i + 1);
                continue;
            } else {
                sum += arr[i];
                sum -= arr[i - k];
                double temp = sum / k;
                ans = Math.max(ans, temp/avg);
                avg = temp;
            }
        }
        return ans;
    }
}
