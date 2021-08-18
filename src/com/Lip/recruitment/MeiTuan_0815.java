package com.Lip.recruitment;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

public class MeiTuan_0815 {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int a = sc.nextInt();
        int d = sc.nextInt();
        int c2 = sc.nextInt();
        int c3 = sc.nextInt();
        int c4 = sc.nextInt();
        String str = sc.next();

        System.out.println(solution_5(n, a, d, c2, c3, c4, str));
    }

    private static long solution_5(int n, int a, int d, int c2, int c3, int c4, String str) {
        long ans = 0;
        char[] chs = str.toCharArray();
        int td = 0;
        int[] ta = new int[n];
        Arrays.fill(ta, 1);

        for (int i = 0; i < n; i++) {
            if (td != 0) td--;
            if (chs[i] == '1') {
                if (td != 0) {
                    ans += (a * ta[i]);
                } else {
                    ans += (a * ta[i] - d) > 0 ? (a * ta[i] - d) : 1;
                }
            } else if (chs[i] == '2') {
                for (int j = i + 1; j <= i + c2 && j < n; j++) {
                    ta[j]++;
                }
            } else if (chs[i] == '3') {
                td = c3 + 1;
            } else {
                a += c4;
            }
        }

        return ans;
    }


    public static int solution_4(String s,int start,int end){
        int ans=0;
        for (int i = start; i <end ; i++) {
            int endTemp=end,j=i,ansTemp=0;
            while ( (j+1)<=endTemp || j<=endTemp){
                if (s.charAt(j)!=s.charAt(endTemp)){
                    ansTemp=0;
                    break;
                }
                ansTemp+=2;
                endTemp--;j++;
            }

            if (j-1==endTemp+1){
                ansTemp--;
            }
            ans=Math.max(ansTemp,ans);
        }
        return ans;

    }


}
