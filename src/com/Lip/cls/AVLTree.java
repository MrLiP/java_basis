package com.Lip.cls;


// 首先定义树节点
class AVLTreeNode{
    private int data;
    private int height;
    private AVLTreeNode left;
    private AVLTreeNode right;
    public int getData()
    { return data; }
    public void setData(int data)
    { this.data=data;}
    public int getHeight()
    { return height; }
    public void setHeight(int height)
    { this.height=height; }
    public AVLTreeNode getLeft()
    { return left; }
    public void setLeft(AVLTreeNode left)
    { this.left=left; }
    public AVLTreeNode getRight()
    { return right; }
    public void setRight(AVLTreeNode right)
    { this.right=right; }
}

class MyAVL{
    public int Height(AVLTreeNode root) {
        if(root == null)
            return -1;
        else
            return root.getHeight();
    }

    // 1.左左插入  左向右旋转
    private AVLTreeNode SingleRotataLeft(AVLTreeNode X) {
        AVLTreeNode W = X.getLeft();
        X.setLeft(W.getRight());
        W.setRight(X);

        X.setHeight(Math.max(Height(X.getLeft()), Height(X.getRight())) + 1);
        W.setHeight(Math.max(Height(W.getLeft()), X.getHeight()) + 1);

        return W;
    }

    //2. 右右插入  右向左旋转
    private AVLTreeNode SingleRotataRight(AVLTreeNode X) {
        AVLTreeNode W = X.getRight();
        X.setRight(W.getLeft());
        W.setLeft(X);

        X.setHeight(Math.max(Height(X.getRight()), Height(X.getLeft()))+1);
        W.setHeight(Math.max(Height(W.getRight()), X.getHeight())+1);

        return W;
    }

    //左右插入  先将下一个节点整体向左旋转，然后向右旋转
    private AVLTreeNode DoubleRotatewithLeft(AVLTreeNode Z) {
        Z.setLeft(SingleRotataRight(Z.getLeft()));
        return SingleRotataLeft(Z);
    }

    //右左插入  先将下一个节点整体向右旋转，然后向左旋转
    private AVLTreeNode DoubleRotatewithRight(AVLTreeNode Z) {
        Z.setRight(SingleRotataLeft(Z.getRight()));
        return SingleRotataRight(Z);
    }


    //插入操作
    public AVLTreeNode Insert(AVLTreeNode root,AVLTreeNode parent,int data) {
        if(root==null) {
            root = new AVLTreeNode();
            root.setData(data);
            root.setHeight(0);
            root.setLeft(null);
            root.setRight(null);
        }
        else if(data < root.getData()) {
            root.setLeft(Insert(root.getLeft(), root, data));
            if(Height(root.getLeft()) - Height(root.getRight()) == 2) {
                if(data < root.getLeft().getData())
                    root = SingleRotataLeft(root);
                else
                    root = DoubleRotatewithLeft(root);
            }
        }
        else if(data > root.getData()) {
            root.setRight(Insert(root.getRight(), root, data));
            if(Height(root.getRight()) - Height(root.getLeft()) == 2) {
                if(data < root.getRight().getData())
                    root = SingleRotataRight(root);
                else
                    root = DoubleRotatewithRight(root);
            }
        }
        root.setHeight(Math.max(Height(root.getLeft()), Height(root.getRight()))+1);
        return root;
    }

    //遍历操作：
    public void dfs(AVLTreeNode first)//中序遍历
    {
        if(first!=null)
        {
            dfs(first.getLeft());
            System.out.print(first.getData());
            dfs((first.getRight()));
        }
    }
}

public class AVLTree {
    public static void main(String[] args) {
        AVLTreeNode one=new AVLTreeNode();
        one.setData(6);
        MyAVL text = new MyAVL();
        text.Insert(one,null,5);
        text.Insert(one,null,9);
        text.Insert(one,null,7);
        text.Insert(one,null,8);
        text.Insert(one,null,3);
        text.dfs(one);
    }
}