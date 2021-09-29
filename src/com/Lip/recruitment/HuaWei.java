package com.Lip.recruitment;

import java.util.Scanner;

public class HuaWei {

    private static int solution_1(String text1, String text2) {
        int[][]dp=new int[text1.length()+1][text2.length()+1];
        for(int i=1;i<=text1.length();i++){
            for(int j=1;j<=text2.length();j++){
                if(text1.charAt(i-1)==text2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1]+1;
                }else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        //List<Integer> list = new ArrayList<Integer>;

        int[] nums = new int[10];
        for(int i = 0; i < 10; i++) {
            nums[i] = sc.nextInt();
        }

        nums = solution_2(nums);

        for (int i = 0; i < 10; i++) {
            System.out.print(nums[i] + " ");
        }
    }

    private static int[] solution_2(int[] nums) {
        int p1 = -1, p2 = -1;

        for ( int i = 0; i < nums.length; i++) {
            if (nums[i] < 3) {
                p2++;
                swap(i, p2, nums);

                if (nums[p2] < 2) {
                    p1++;
                    swap(p1, p2, nums);
                }
            }
        }
        return nums;
    }

    private static void swap(int a, int b, int[] nums) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }
}


