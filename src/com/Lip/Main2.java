package com.Lip;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main2 {

    public static void main(String[] args) {


    }

    public static Integer[] solution(int[] a, int[] b) {
        ArrayList<Integer> ans = new ArrayList<>();

        int l = 0, r = 0;

        while (l < a.length && r < b.length) {
            if (a[l] == b[r]) {
                ans.add(a[l]);
                l++;
                r++;
            } else if (a[l] > b[r]) {
                r++;
            } else {
                l++;
            }
        }

        return ans.toArray(new Integer[0]);
    }

}
