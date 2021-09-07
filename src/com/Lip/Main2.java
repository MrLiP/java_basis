package com.Lip;

import java.util.*;

public class Main2 {

    static long ans = 0;

    public static void main(String[] args) {


        System.out.println(GetNumOfExpress("1^0|0|1", false));
    }


    /**
     * Note: 类名、方法名、参数名已经指定，请勿修改
     *
     *
     * 计算express结果值满足desired的组合种类数
     * @param express string字符串 express字符串
     * @param desired bool布尔型 预期结果
     * @return long长整型
     */
    public static long GetNumOfExpress(String express, boolean desired) {
        // write code here

        //dfs(express, desired);

        char[] chs = express.toCharArray();

        //int[][] t = new int[chs.length][chs.length];
        //int[][] f = new int[chs.length][chs.length];



        return dfs(chs, 0, chs.length - 1, desired);
    }

    private static int dfs(char[] chs, int start, int end, boolean desired){
        //递归结束条件
        if(start == end){
            if((chs[start] == '1' && desired) ||  (chs[start] == '0' && !desired)){
                return 1;
            }else {
                return 0;
            }
        }

        int count = 0;

        for(int i = start+1; i < end; i += 2){
            if(chs[i] == '&' && desired){
                count += dfs(chs,start,i-1,true) * dfs(chs,i+1,end,true);
            }else if((chs[i] == '&' && !desired)){
                count += dfs(chs,start,i-1,true) * dfs(chs,i+1,end,false)
                        + dfs(chs,start,i-1,false) * dfs(chs,i+1,end,true)
                        + dfs(chs,start,i-1,false) * dfs(chs,i+1,end,false);
            }else if((chs[i] == '|' && desired)){
                count += dfs(chs,start,i-1,true) * dfs(chs,i+1,end,false)
                        + dfs(chs,start,i-1,false) * dfs(chs,i+1,end,true)
                        + dfs(chs,start,i-1,true) * dfs(chs,i+1,end,true);
            }else if(chs[i] == '|' && !desired){
                count += dfs(chs,start,i-1,false) * dfs(chs,i+1,end,false);
            }else if(chs[i] == '^' && desired){
                count += dfs(chs,start,i-1,true) * dfs(chs,i+1,end,false)
                        + dfs(chs,start,i-1,false) * dfs(chs,i+1,end,true);
            }
            else if(chs[i] == '^' && !desired){
                count += dfs(chs,start,i-1,true) * dfs(chs,i+1,end,true)
                        + dfs(chs,start,i-1,false) * dfs(chs,i+1,end,false);
            }
        }
        return count;
    }
}
