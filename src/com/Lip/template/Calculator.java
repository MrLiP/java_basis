package com.Lip.template;

import java.util.*;

// 计算表达式
public class Calculator {
    public static void main(String[] args){
        Scanner sc =  new Scanner(System.in);
        while(sc.hasNext()){
            //在连续输入如果中间含有空格，那就用nextLine()，不然就用next()
            String s = sc.nextLine();
            char[] charArray = s.toCharArray();
            LinkedList<Character> list = new LinkedList<>();
            for (char c : charArray) {
                list.offer(c);
            }
            System.out.println(calculate(list));
        }
    }
    public static int calculate(LinkedList<Character> list){
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        int pre = 0;
        int num = 0;
        while(!list.isEmpty()){
            char temp = list.poll();  // queue
            if(temp=='('){
                num =calculate(list);
            }
            if(temp>='0'&&temp<='9'){
                num = num*10+(temp-'0');
            }
            //temp<'0'||temp>'9'||list.isEmpty() 这个保证最后一个也能进来
            if((temp<'0'||temp>'9'||list.isEmpty())&&temp!=' '){
                if(sign=='+'){
                    stack.push(num);
                }
                if(sign=='-'){
                    stack.push(-num);
                }
                if(sign=='*'){
                    pre = stack.pop();
                    stack.push(pre*num);
                }
                if(sign=='/'){
                    pre = stack.pop();
                    stack.push(pre/num);
                }
                sign = temp;
                num = 0;
            }
            if(temp==')'){
                break;
            }
        }
        int res = 0;
        while(!stack.isEmpty()){
            res+=stack.pop();
        }
        return res;
    }
}
