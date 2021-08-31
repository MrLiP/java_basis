package com.Lip;

import java.util.*;


public class Main {

    public static int maximal_square (String[][] matrix) {
        int ans = 0;
        if (matrix.length == 0 || matrix[0].length == 0) return ans;

        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == "1") {
                    if (i == 0 || j == 0) dp[i][j] = 1;
                    else {
                        int temp = Math.min(dp[i - 1][j], dp[i][j - 1]);
                        dp[i][j] = Math.min(temp, dp[i - 1][j - 1]) + 1;
                    }
                    ans = Math.max(ans, dp[i][j]);
                }
            }
        }

        return ans;
    }

    static HashMap<Character, Integer> map = new HashMap<Character, Integer>(){{
        put('A', 1);
        put('J', 11);
        put('Q', 12);
        put('K', 13);
    }};

    public static boolean hasStraightFlush (String[] cards) {
        int len = cards.length, r, c;
        if (len < 5) return false;
        int[][] arr = new int[4][13];

        for (int i = 0; i < len; i++) {
            char color = cards[i].charAt(0);
            if (color == 'S') r = 0;
            else if (color == 'H') r = 1;
            else if (color == 'C') r = 2;
            else r = 3;

            char ch = cards[i].charAt(1);
            if (map.containsKey(ch)) {
                c = map.get(ch);
            } else {
                if (cards[i].length() < 3) c = ch - '0';
                else c = 10;
            }
            arr[r][c - 1] += 1;
        }

        for (int i = 0; i < 4; i++) {
            int index = 0;
            for (int j = 0; j < 13; j++) {
                if (arr[i][j] != 0) index++;
                else index = 0;
                if (index == 5) return true;
            }
            index = 0;
            for (int j = 0; j < 5; j++) {
                if (arr[i][(9 + j) % 13] != 0) index++;
            }
            if (index == 5) return true;
        }
        return false;
    }

    public static void main(String[] args) {
        String[][] input1 = new String[][]{{"1", "1", "1"}, {"1", "1", "1"}};
        String[] input2 = new String[]{"H3", "H4", "H6", "H7", "H8", "H9"};

        System.out.println(hasStraightFlush(input2));

    }



}
