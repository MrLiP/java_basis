package com.Lip;

import java.text.DecimalFormat;
import java.util.*;

public class Main1 {
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

        Scanner sc = new Scanner(System.in);
        String[] str = sc.nextLine().split(",");
        String[] lk = str[str.length - 1].split(":");
        int[] arr = new int[str.length];
        for (int i = 0; i < arr.length - 1; i++) {
            arr[i] = Integer.parseInt(str[i]);
        }
        arr[arr.length - 1] = Integer.parseInt(lk[0]);
        int k = Integer.parseInt(lk[1]);

//        System.out.println(solution(arr, k));
        DecimalFormat df = new DecimalFormat("0.00%");
        String r = df.format(solution(arr, k));
        System.out.println(r);

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
