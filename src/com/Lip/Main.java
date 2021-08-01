package com.Lip;

import java.util.Scanner;
import java.util.Stack;


public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String str = sc.nextLine();
        System.out.println(solution_1(str));

//        System.out.print(solution("[][[]2]2"));
    }

    // 数箱子，[][[][][]2]3 >> 16
    private static int solution_1(String str) {
        int ans = 0;
//        Stack<Character> chs = new Stack<Character>();
        Stack<Integer> stack = new Stack<>();
        stack.add(0);

        char[] charr = str.toCharArray();

        for (int i = 0; i < charr.length; i++) {
            char ch = str.charAt(i);

            if (ch == '[') {
//                chs.push(ch);
                stack.push(1);
            } else if (ch == ']') {
                if (i < charr.length - 1 && str.charAt(i + 1) > '0' && str.charAt(i + 1) <= '9') {
                    int a = stack.pop();
                    int multi = str.charAt(i + 1) - '0';
                    stack.push(a * multi);
                    i++;
                }
                int fst = stack.pop();
                int snd = stack.pop();
                stack.push(fst + snd);
            }
        }
        int num = stack.size();
        for (int i = 0; i < num; i++) {
            ans += stack.pop();
        }

        return ans;
    }
}
