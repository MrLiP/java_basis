package com.Lip.cls;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class YingJiao {
    static int temp = 0;

    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int[] a1 = new int[]{5,4,3,2,1};
//        int[] a2 = new int[]{1,2,3,4,5};

        int[] a = new int[]{1,2,3,4};
        int[] t = new int[]{0,0,1,2,1,2};
        int[] c = new int[]{0,0,1,1,1,0};
        int[] d = new int[]{1,1,2,0,2,1};

        // [1,2,3],[0,0,2],[0,0,1],[1,2,0]
//        int[] a = new int[]{1,2,3};
//        int[] t = new int[]{0,0,2};
//        int[] c = new int[]{0,0,1};
//        int[] d = new int[]{1,2,0};

//        System.out.println(increaseInterval(a2));
//        System.out.println(step(7));
        System.out.println(Arrays.toString(work(a, t, c, d)));
    }

    public static int[] work (int[] a, int[] t, int[] c, int[] d) {
        // write code here
        int n = a.length, m = t.length, count = 0;
        for (int value : t) {
            if (value == 2) count++;
        }
        int[] work = new int[count];
        if (count == 0) return work;
        count = 0;

        // map1:气球 对应 捆数
        HashMap<Integer, Integer> map1 = new HashMap<>();
        // map2：捆数 对应 气球
        HashMap<Integer, ArrayList<Integer>> map2 = new HashMap<>();

        for (int i = 0; i < n; i++) {
            map1.put(i, i);
            //map1.get(i).add(i);
            map2.put(i, new ArrayList<>());
            map2.get(i).add(i);
        }

        ArrayList<Integer> list1, list2;
        for (int i = 0; i < m; i++) {
            if (t[i] == 0) {
                int t1 = c[i], t2 = d[i];
                if (!map1.get(t1).equals(map1.get(t2))) {
                    //int t3 = map1.get(t2);

                    list1 = map2.get(map1.get(t1));
                    list2 = map2.remove(map1.get(t2));
                    list1.addAll(list2);

                    map1.replace(t2, map1.get(t1));
                }
            } else if (t[i] == 1) {
                list1 = map2.get(map1.get(c[i]));
                for (int num : list1) {
                    a[num] += d[i];
                }
            } else {
                work[count++] = a[c[i]];
            }
        }

        return work;
    }

    public static int step (int n) {
        // write code here
//        int[][] dp = new int[n + 1][2 * n + 1];
//        dp[0][0] = 1;

        int[] a = new int[2 * n + 1];
        a[0] = 1;

        for (int i = 0; i < n + 1; i++) {
            int[] b = new int[2 * n + 1];

            for (int j = 0; j <= 2 * i; j++) {
                if (j == 0) {
                    b[j] = 1;
                } else {
                    b[j] = a[j] + b[j - 1];
                    if (b[j] >= 10000) b[j] %= 10000;
                }

            }
            a = b;
        }

        return a[2 * n];
    }

    public static int increaseInterval (int[] a) {
        // write code here
        int ans = 1;
        int temp = a[0];
        int sum = 0;

        for (int i = 1; i < a.length; i++) {
            sum += a[i];
            if (sum > temp) {
                ans++;
                temp = sum;
                sum = 0;
            }
        }

        return ans;
    }
}
