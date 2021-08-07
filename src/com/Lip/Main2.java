package com.Lip;

import java.text.DecimalFormat;
import java.util.Scanner;

public class Main2 {
    private static int x = 100;
    public static void main(String[] args) {
        Main2 hs1 = new Main2();
        hs1.x++;
        Main2 hs2 = new Main2();
        hs2.x++;
        hs1 = new Main2();
        hs1.x++;
        Main2.x--;
        System.out.println("x="+x);

    }

}
