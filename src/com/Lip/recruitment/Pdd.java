package com.Lip.recruitment;

import com.Lip.TreeNode;

import java.util.*;

public class Pdd {

    static Scanner sc = new Scanner(System.in);

    public static void solution_1() {

        int N = Integer.parseInt(sc.next());
        for (int i = 0; i <N ; i++) {
            int x = Integer.parseInt(sc.next());
            int y = Integer.parseInt(sc.next());
            int r = Integer.parseInt(sc.next());
            x = Math.abs(x);
            y = Math.abs(y);
            int rr = x*x+y*y;
            int count1=0,count2 = 0;
            if (x<r){
                count1 = 2;
            }else if (x==r){
                count1 = 1;
            }
            if (y<r){
                count2 = 2;
            }else if (y==r){
                count2 = 1;
            }
            if (rr==r*r){
                System.out.println(count1+count2-1);
            }else{
                System.out.println(count1+count2);
            }
        }
    }

    public static void solution_2() {

        int n = sc.nextInt();//n个糖果
        int t = sc.nextInt();//多多能够接受的糖果的甜度
        int c = sc.nextInt();//多多准备选择的糖果的个数
        int[] candy = new int[n];
        for (int i = 0; i < n; i++) {
            candy[i] = sc.nextInt();
        }
        int res = 0;
        int left = 0;
        int right = 0;
        while (right<n){
            if ((right-left)<c){
                if (candy[right]<=t){
                    if ((right-left)==c-1) res++;
                }else {
                    left = right+1;
                }
                right++;
            }else {
                left++;
            }
        }
        System.out.println(res);
    }


    public static int[] solution_3(int n, String str) {
        int cur = 0, length = 0;
        char[] chs = str.toCharArray();
        StringBuilder sb = new StringBuilder();
        int[] ans = new int[n];

        for (int i = 0; i < n; i++) {
            if (chs[i] == 'L') {
                cur = cur > 0 ? cur - 1 : 0;
            } else if (chs[i] == 'R') {
                cur = cur < length ? cur + 1 : cur;
            } else if (chs[i] == 'D') {
                if (cur != 0){
                    sb.deleteCharAt(cur - 1);
                    length--;
                    cur--;
                }
            } else {
                sb.insert(cur, chs[i]);
                cur++;
                length++;
            }
            ans[i] = score(sb.toString());
        }
        return ans;
    }

    public static int score(String str) {
        int depth = 0;
        if (str.length() == 0) return depth;

        if (isValid(str)) {
            int temp = 0;
            for (int i = 0; i < str.length(); i++) {
                char ch = str.charAt(i);
                if (ch == '(') {
                    temp++;
                    depth = Math.max(depth, temp);
                }
                else temp--;
            }
            return depth;
        } else {
            int bal = 0;
            for (int i = 0; i < str.length(); i++) {
                bal += str.charAt(i) == '(' ? 1 : -1;
                if (bal == -1) {
                    depth++;
                    bal++;
                }
            }
            return -(depth + bal);
        }
    }

    public static boolean isValid(String str){
        int n = str.length();
        if (n % 2 != 0) return false;

        Map<Character, Character> pairs = new HashMap<>();
        pairs.put(')', '(');

        Deque<Character> stack = new LinkedList<>();

        for (int i = 0; i < n; i++) {
            char ch = str.charAt(i);
            if (pairs.containsKey(ch)) {
                if (stack.isEmpty()) return false;
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }

    public static void solution_4() {

        int n = sc.nextInt();//表示总共有多少道题
        int m = sc.nextInt();//表示前面一题的难度只能比后面一题的难度多m
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = sc.nextInt();
        }
        Arrays.sort(nums);
        int res = 1;//存储最终结果
        int k ;//存储每个情况的种类
        int[] arr = new int[2];
        arr[0] = 1;//记录次数
        arr[1] = n-1;//记录后一个节点在哪停止
        for (int i = n-1; i > 0; i--) {
            arr = findPossible(nums,i,m,arr[1]);
            res = arr[0]*res;
        }
        System.out.println(res);
    }

    private static int[] findPossible(int[] nums, int i ,int m,int k ) {
        int count = i-k+1;
        for (int j = k-1; j >=0 ; j--) {
            if ((nums[i]-nums[j])<=m) count++;
            else {
                k = j+1;
                break;
            }
        }
        int[] arr = new int[2];
        arr[0] = count;
        arr[1] = k;
        return arr;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        sc.nextLine();
        String str = sc.nextLine();

        int[] ans = solution_3(n, str);

        for (int i : ans) {
            System.out.print(i + " ");
        }
    }

}
