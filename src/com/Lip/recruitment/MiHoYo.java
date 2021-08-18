package com.Lip.recruitment;


import java.util.*;

public class MiHoYo {

    static class P{
        int x;
        int y;

        public P(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        int t = scanner.nextInt();
        while(t>0){
            int n = scanner.nextInt();
            int m = scanner.nextInt();
            int x1 = scanner.nextInt()-1;
            int y1 = scanner.nextInt()-1;
            int x2 = scanner.nextInt()-1;
            int y2 = scanner.nextInt()-1;
            char[][] map = new char[n][m];
            for(int i = 0; i < n; i++){
                String s = scanner.next();
                for(int j = 0; j < m; j++)
                    map[i][j] = s.charAt(j);
            }

            int step = 0;
            List<P> list = new ArrayList<>();
            list.add(new P(x1, y1));
            Set<String> set = new HashSet<>();
            int flag = 0;

            while(flag == 0){
                List<P> list2 = new ArrayList<>();
                if(list.size() == 0){
                    flag = 1;
                    continue;
                }
                for(P p : list){
                    // 缓存
                    if(set.contains(p.x + "," + p.y))
                        continue;
                    set.add(p.x + "," + p.y);
                    // 检查是否到终点
                    if(reach(p.x, p.y, x2, y2)){
                        flag = 2;
                        break;
                    }
                    //扩展
                    if(p.x+1<n&&map[p.x+1][p.y]=='.')
                        list2.add(new P(p.x+1,p.y));
                    if(p.y+1<m&&map[p.x][p.y+1]=='.')
                        list2.add(new P(p.x,p.y+1));
                    if(p.x-1>=0&&map[p.x-1][p.y]=='.')
                        list2.add(new P(p.x-1,p.y));
                    if(p.y-1>=0&&map[p.x][p.y-1]=='.')
                        list2.add(new P(p.x,p.y-1));

                }
                list = list2;
                if(flag != 2)
                    step++;
            }
            if(flag == 1)
                System.out.println(-1);
            else{
                x1++;
                y1++;
                x2++;
                y2++;
                System.out.println(step ^ (x1 * x2) ^ (y1 * y2));
            }
            t--;
        }
    }

    public static boolean reach(int x,int y,int x2,int y2){
        int t1 = Math.abs(x-x2);
        int t2 =  Math.abs(y-y2);
        return t1 <= 1 && t2 <= 1;
    }
}
