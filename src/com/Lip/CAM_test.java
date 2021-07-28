package com.Lip;

import java.util.Scanner;

public class CAM_test {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while(in.hasNext()){
            int k=in.nextInt();
            int a=in.nextInt();
            int b=in.nextInt();
            System.out.println(getDay(k,a,b));
        }
    }
    public static int getDay(int k,int a,int b){
        while (a!=b){
            int n=(k-a)/(a-b);
            int m=(k-a)%(a-b);
            if (k>0 && a>k){
                return 1;
            }
            if (m==0){
                return n+1;
            }
            else {
                return n+2;
            }
        }
        if (k>0 && k>a && a<=b){
            return -1;
        }
        if (k<0 && a>=b){
            return -1;
        }
        return -1;
    }
}
