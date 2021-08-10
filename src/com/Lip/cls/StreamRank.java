package com.Lip.cls;

import java.util.ArrayList;
import java.util.List;

class StreamRank {

    public StreamRank() {
        list = new ArrayList<>();
    }
    List<Integer> list;
    public void track(int x) {
        int i = 0;
        int j = list.size()-1;
        while(i <= j){
            int m = i + (j-i)/2;
            if(list.get(m) < x){i = m+1;}
            else{j = m-1;}
        }// 寻找第一个比它大的数
        list.add(i, x);// 在其下标处插入x
    }

    public int getRankOfNumber(int x) {
        int i = 0;
        int j = list.size()-1;
        while(i <= j){
            int m = i + (j-i)/2;
            if(list.get(m) > x){j = m-1;}
            else{i = m+1;}
        }// 寻找最后一个不大于它的数
        return j+1;// 下标加1才是元素数目（秩）
    }
}
