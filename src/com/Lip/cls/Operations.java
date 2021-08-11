package com.Lip.cls;

public class Operations {
    int[] postiveBit; //二进制值数组 正值
    int[] negBit; //二进制数组 负值
    public Operations() {
        int sum=0;
        int sum2=0;
        postiveBit=new int[31];
        negBit=new int[31];
        for(int i=0;i<31;i++){
            postiveBit[i]=sum+1;
            sum+=postiveBit[i];
            negBit[i]=sum2+0xFFFFFFFF;
            sum2+=negBit[i];
        }
    }

    public int minus(int a, int b) {
        long tmpB=Math.abs(b);
        long ans=a;
        for(int i=30;i>=0&&tmpB!=0;i=i+0xFFFFFFFF){ //这里不能用 i-- 呜呜呜
            if(tmpB>=postiveBit[i]){
                tmpB+=negBit[i];
                if(b>0){
                    ans+=negBit[i];
                }else{
                    ans+=postiveBit[i];
                }
            }
        }
        return (int) ans;
    }

    public int multiply(int a, int b) {
        long ans=0;
        int tmpA=Math.abs(a);
        boolean[] arr=new boolean[31];
        for(int i=30;i>=0&&tmpA!=0;i=i+0xFFFFFFFF){
            if(tmpA>=postiveBit[i]){
                tmpA+=negBit[i];
                arr[i]=true;
            }
        }
        long sum=b;
        for(int i=0;i<31;i++){
            if(arr[i]){
                ans+=sum;
            }
            sum+=sum;
        }
        return a>0?(int)ans:minus(0,(int)ans);
    }
    public int divide(int a, int b) {
        if(b==1){
            return a;
        }
        boolean op=false;
        if(a>0&&b>0||a<0&&b<0){
            op=true;
        }
        long tmpA=Math.abs(a);
        long tmpB=Math.abs(b);
        long ans=0;
        for(int i=30;i>=0&&tmpA>=tmpB;i=i+0xFFFFFFFF){
            if(tmpA>=tmpB*postiveBit[i]){
                tmpA+=tmpB*negBit[i];
                ans+=postiveBit[i];
            }
        }
        return op?(int)ans:minus(0,(int)ans);
    }
}


/**
 * Your Operations object will be instantiated and called as such:
 * Operations obj = new Operations();
 * int param_1 = obj.minus(a,b);
 * int param_2 = obj.multiply(a,b);
 * int param_3 = obj.divide(a,b);
 */
