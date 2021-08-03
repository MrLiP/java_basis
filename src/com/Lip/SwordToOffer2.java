package com.Lip;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.text.CollationElementIterator;
import java.util.*;

class SolutionSTO{
    // 剑指 Offer 03. 数组中重复的数字  easy
    // HashSet / 原地交换(值-索引)，两者是 多对一 的关系
    public int findRepeatNumber(int[] nums) {
        Set<Integer> dic = new HashSet<>();

        for (int num : nums) {
            if (dic.contains(num)) return num;
            dic.add(num);
        }
        return 0;
    }

    // 剑指 Offer 04. 二维数组中的查找  medium
    // 对角线二分
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while(i >= 0 && j < matrix[0].length) {
            if (target == matrix[i][j]) return true;
            else if (target > matrix[i][j]) j++;
            else i--;
        }
        return false;
    }

    // 剑指 Offer 05. 替换空格  easy
    // StringBuilder  线程不安全，StringBuffer 线程安全
    // for (Character c : s.toCharArray())
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder();
        for (Character c : s.toCharArray()) {
            if (c == ' ') res.append("%20");
            else res.append(c);
        }
        return res.toString();
    }

    // 剑指 Offer 06. 从尾到头打印链表  easy
    // Stack reverse, addLast() removeLast()
    public int[] reversePrint(ListNode head) {
        LinkedList<Integer> stack = new LinkedList<>();
        while (head != null) {
            stack.addLast(head.val);
            head = head.next;
        }
        int[] res = new int[stack.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = stack.removeLast();
        }
        return res;
    }

    // 剑指 Offer 07. 重建二叉树  medium
    // 递归参数：根节点在前序遍历的索引 root 、子树在中序遍历的左边界 left 、子树在中序遍历的右边界 right
    int[] preorder_07;
    HashMap<Integer, Integer> dic_07 = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder_07 = preorder;
        for(int i = 0; i < inorder.length; i++)
            dic_07.put(inorder[i], i);
        return recur(0, 0, inorder.length - 1);
    }

    TreeNode recur(int root, int left, int right) {
        if (left > right) return null;                          // 递归终止
        TreeNode node = new TreeNode(preorder_07[root]);          // 建立根节点
        int i = dic_07.get(preorder_07[root]);                       // 划分根节点、左子树、右子树
        node.left = recur(root + 1, left, i - 1);              // 开启左子树递归
        node.right = recur(root + i - left + 1, i + 1, right); // 开启右子树递归，此处重复了一个 left 故减去
        return node;
    }

    // 剑指 Offer 09. 用两个栈实现队列  easy
    // LinkedList, addLast(), removeLast(), 来回倒腾

    // 剑指 Offer 10- I. 斐波那契数列  easy
    public int fib(int n) {
        if (n == 0 || n == 1) return n;
        int a = 0, b = 1, c = 0;
        for (int i = 2; i <= n; i++) {
            c = (a + b) % (1000000007);
            a = b;
            b = c;
        }
        return c ;
    }

    // 剑指 Offer 10- II. 青蛙跳台阶问题  easy
    public int numWays(int n) {
        int a = 1, b = 1;
        for (int i = 0; i < n; i++) {
            int sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }

    // 剑指 Offer 11. 旋转数组的最小数字  easy
    // 二分法 确保最小值的区间，注意 if (numbers[mid] == numbers[right]) right-- 操作
    public int minArray(@NotNull int[] numbers) {
        int l = numbers.length, left = 0, right = l - 1;

        while (left < right) {
            int mid = (left + right) >> 1;
            if (numbers[mid] > numbers[right]) {
                left = mid + 1;
            } else if (numbers[mid] < numbers[right]) {
                right = mid;
            } else {
                right--;
            }
        }
        return numbers[left];
    }

    // 剑指 Offer 12. 矩阵中的路径  medium
    // dfs + 剪纸，利用源数组直接赋值，节省 visited[][] 的空间，利用 || 操作
    public boolean exist(char[][] board, String word) {
        int m = board.length, n = board[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (check(board, i, j, word, 0)) return true;
                }
            }
        }
        return false;
    }

    boolean check(char[][] board, int i, int j, String word, int index) {
        if(i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != word.charAt(index)) return false;
        if(index == word.length() - 1) return true;
        board[i][j] = '\0';
        boolean res = check(board, i + 1, j, word,index + 1) || check(board, i - 1, j, word,index + 1) ||
                check(board, i, j + 1, word,index + 1) || check(board, i , j - 1, word,index + 1);
        board[i][j] = word.charAt(index);
        return res;
    }

    // 剑指 Offer 13. 机器人的运动范围  medium
    // BFS, Queue, offer(), poll(), vis[][]
    public int movingCount(int m, int n, int k) {
        if (k == 0) return 1;
        Queue<int[]> queue = new LinkedList<>();
        // 向右和向下的方向数组
        int[] dx = {0, 1};
        int[] dy = {1, 0};
        boolean[][] vis = new boolean[m][n];
        queue.offer(new int[]{0, 0});
        vis[0][0] = true;
        int ans = 1;
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0], y = cell[1];
            for (int i = 0; i < 2; i++) {
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                if (tx < 0 || tx >= m || ty < 0 || ty >= n || vis[tx][ty] || get(tx) + get(ty) > k) {
                    continue;
                }
                queue.offer(new int[]{tx, ty});
                vis[tx][ty] = true;
                ans++;
            }
        }
        return ans;
    }

    private int get(int x) {
        int res = 0;
        while (x != 0) {
            res += x % 10;
            x /= 10;
        }
        return res;
    }

    // 剑指 Offer 14- I. 剪绳子  medium
    // 数学推导，等分切，最优切三段
    public int cuttingRope_1(int n) {
        if(n <= 3) return n - 1;
        int a = n / 3, b = n % 3;
        if(b == 0) return (int)Math.pow(3, a);
        if(b == 1) return (int)Math.pow(3, a - 1) * 4;
        return (int)Math.pow(3, a) * 2;
    }

    // 剑指 Offer 14- II. 剪绳子 II  medium
    // 循环求余，long res = 1L, int p = (int)1e9+7;
    public int cuttingRope_2(int n) {
        if(n <= 3) return n - 1;
        long res = 1L;
        int p = (int)1e9+7;
        //贪心算法，优先切三，其次切二
        while(n>4){
            res = res*3%p;
            n -= 3;
        }
        //出来循环只有三种情况，分别是n=2、3、4
        return (int)(res*n%p);
    }

    // 剑指 Offer 15. 二进制中1的个数  easy
    // 位运算，>>>, 无符号移位
    public int hammingWeight(int n) {
        int ans = 0;
        while (n != 0) {
            if ((n & 1) == 1) ans++;
            n >>>= 1;
        }
        return ans;
    }

    // 剑指 Offer 16. 数值的整数次方  medium
    // 快速幂, 幂的二进制展开 x^n = x^{1*b_1}x^{2*b_2}x^{4*b_3}...x^{2^{m-1}*b_m}x
    public double myPow(double x, int n) {
        if (x == 0) return 0;
        long b = n;
        double res = 1.0;
        if (b < 0) {
            x = 1 / x;
            b = - b;
        }
        while (b > 0) {
            if ((b & 1) == 1) res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }

    // 剑指 Offer 17. 打印从1到最大的n位数  easy
    // 针对大数问题，使用 String
    public int[] printNumbers(int n) {
        int l = (int)(Math.pow(10, n) - 1);
        int[] res = new int[l];
        for (int i = 0; i < l; i++) {
            res[i] = i + 1;
        }
        return res;
    }

    // 考虑大数问题
//    class Solution {
//        StringBuilder res;
//        int nine = 0, count = 0, start, n;
//        char[] num, loop = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
//        public String printNumbers(int n) {
//            this.n = n;
//            res = new StringBuilder();
//            num = new char[n];
//            start = n - 1;
//            dfs(0);
//            res.deleteCharAt(res.length() - 1);
//            return res.toString();
//        }
//        void dfs(int x) {
//            if(x == n) {
//                String s = String.valueOf(num).substring(start);
//                if(!s.equals("0")) res.append(s + ",");
//                if(n - start == nine) start--;
//                return;
//            }
//            for(char i : loop) {
//                if(i == '9') nine++;
//                num[x] = i;
//                dfs(x + 1);
//            }
//            nine--;
//        }
//    }

    // 剑指 Offer 18. 删除链表的节点  easy
    // dummy, pre, cur
    public ListNode deleteNode(ListNode head, int val) {
        if (head.val == val) return head.next;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        while (head.next.val != val) {
            head = head.next;
        }
        head.next = head.next.next;
        return dummy.next;
    }

    // 剑指 Offer 19. 正则表达式匹配  hard
    // dp[][], 初始化首行，分情况讨论（*, else）
    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        // 初始化首行
        for(int j = 2; j <= p.length(); j += 2)
            dp[0][j] = dp[0][j - 2] && p.charAt(j - 1) == '*';

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= p.length(); j++) {
                if(p.charAt(j - 1) == '*') {
                    if(dp[i][j - 2]) dp[i][j] = true;                                            // 1.
                    else if(dp[i - 1][j] && s.charAt(i - 1) == p.charAt(j - 2)) dp[i][j] = true; // 2.
                    else if(dp[i - 1][j] && p.charAt(j - 2) == '.') dp[i][j] = true;             // 3.
                } else {
                    if(dp[i - 1][j - 1] && s.charAt(i - 1) == p.charAt(j - 1)) dp[i][j] = true;  // 1.
                    else if(dp[i - 1][j - 1] && p.charAt(j - 1) == '.') dp[i][j] = true;         // 2.
                }
            }
        }

        return dp[s.length()][p.length()];
    }

    // 剑指 Offer 20. 表示数值的字符串  medium
    // 有限状态自动机 || 依次遍历
    public boolean isNumber(String s) {
        if(s == null || s.length() == 0) return false; // s为空对象或 s长度为0(空字符串)时, 不能表示数值
        boolean isNum = false, isDot = false, ise_or_E = false; // 标记是否遇到数位、小数点、‘e’或'E'
        char[] str = s.trim().toCharArray();  // 删除字符串头尾的空格，转为字符数组，方便遍历判断每个字符
        for(int i=0; i<str.length; i++) {
            if(str[i] >= '0' && str[i] <= '9') isNum = true; // 判断当前字符是否为 0~9 的数位
            else if(str[i] == '.') { // 遇到小数点
                if(isDot || ise_or_E) return false; // 小数点之前可以没有整数，但是不能重复出现小数点、或出现‘e’、'E'
                isDot = true; // 标记已经遇到小数点
            }
            else if(str[i] == 'e' || str[i] == 'E') { // 遇到‘e’或'E'
                if(!isNum || ise_or_E) return false; // ‘e’或'E'前面必须有整数，且前面不能重复出现‘e’或'E'
                ise_or_E = true; // 标记已经遇到‘e’或'E'
                isNum = false; // 重置isNum，因为‘e’或'E'之后也必须接上整数，防止出现 123e或者123e+的非法情况
            }
            else if(str[i] == '-' ||str[i] == '+') {
                if(i!=0 && str[i-1] != 'e' && str[i-1] != 'E') return false; // 正负号只可能出现在第一个位置，或者出现在‘e’或'E'的后面一个位置
            }
            else return false; // 其它情况均为不合法字符
        }
        return isNum;
    }

    // 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面  easy
    // 首尾双指针，快慢双指针
    public int[] exchange(int[] nums) {
        int odd = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] % 2 == 1) {
                if (i == odd) {
                    i++;
                    odd++;
                } else {
                    int temp = nums[i];
                    nums[i] = nums[odd];
                    nums[odd] = temp;
                    odd++;
                }
            }
        }
        return nums;
    }

    // 剑指 Offer 22. 链表中倒数第k个节点  easy
    // 快慢指针
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode fast = head, slow = head;

        for (int i = 0; i < k; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    // 剑指 Offer 24. 反转链表  easy
    // 迭代，递归
    public ListNode reverseList(ListNode head) {
        // 递归
        if (head == null || head.next == null) return head;
        ListNode dummy = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return dummy;

//        迭代
//        ListNode prev = null;
//        ListNode curr = head;
//        while (curr != null) {
//            ListNode next = curr.next;
//            curr.next = prev;
//            prev = curr;
//            curr = next;
//        }
//        return prev;
    }

    // 剑指 Offer 25. 合并两个排序的链表  easy
    // 迭代，递归
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0), cur = dummy;
        while(l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = l1 != null ? l1 : l2;
        return dummy.next;

//        递归
//        if(l1 == null || l2 == null){
//            return l1 != null ? l1 : l2;
//        }
//        if(l1.val <= l2.val){
//            l1.next = mergeTwoLists(l1.next, l2);
//            return l1;
//        }else{
//            l2.next = mergeTwoLists(l1, l2.next);
//            return l2;
//        }
    }

    // 剑指 Offer 26. 树的子结构  medium
    // 递归，先序遍历，check 各个节点
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) return false;
        return check(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    public boolean check(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null) return false;
        return A.val == B.val && check(A.left, B.left) && check(A.right, B.right);
    }

    // 剑指 Offer 27. 二叉树的镜像  easy
    // 递归，迭代
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) return null;
        TreeNode temp = root.left;
//        root.left = root.right;
//        root.right = temp;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(temp);
        return root;

//        迭代
//        if(root == null) return null;
//        Queue<TreeNode> queue = new LinkedList<>();
//        queue.offer(root);
//
//        while(!queue.isEmpty()){
//            TreeNode node = queue.poll();
//            if(node.left != null){
//                queue.offer(node.left);
//            }
//            if(node.right != null){
//                queue.offer(node.right);
//            }
//            TreeNode tmp = node.left;
//            node.left = node.right;
//            node.right = tmp;
//        }
//        return root;
    }

    // 剑指 Offer 28. 对称的二叉树  easy
    // 递归， create recur(TreeNode L, TreeNode R)
    public boolean isSymmetric(TreeNode root) {
        return root == null || recur(root.left, root.right);
    }

    private boolean recur(TreeNode L, TreeNode R) {
        if (L == null && R == null) return true;
        if (L == null || R == null) return false;
        return L.val == R.val && recur(L.left, R.right) && recur(L.right, R.left);
    }

    // 剑指 Offer 29. 顺时针打印矩阵  easy
    // 循环取值，改变 left, right, top, bottom 的边界，利用好 ++i, break
    public int[] spiralOrder(int[][] matrix) {
        if (matrix.length == 0) return new int[0];
        int m = matrix.length, n = matrix[0].length;
        int[] res = new int[m * n];
        int left = 0, right = n-1, top = 0, bottom = m-1, i = 0;
        while(i < m * n) {
            for (int j = left; j <= right; j++) {
                res[i] = matrix[top][j];
                i++;
            }
            if (++top > bottom) break;
            for (int j = top; j <= bottom; j++) {
                res[i] = matrix[j][right];
                i++;
            }
            if (--right < left) break;
            for (int j = right; j >= left; j--) {
                res[i] = matrix[bottom][j];
                i++;
            }
            if (--bottom < top) break;
            for (int j = bottom; j >= top; j--) {
                res[i] = matrix[j][left];
                i++;
            }
            if (++left > right) break;
        }
        return res;
    }

    // 剑指 Offer 30. 包含min函数的栈  easy
    // 建立辅助栈，或者辅助节点 (Go, Node by LinkedList)

    // 剑指 Offer 31. 栈的压入、弹出序列  medium
    // 利用辅助栈，模拟压入弹出的过程, isEmpty(), peek(), push(), pop()
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int value : pushed) {
            stack.push(value);
            while (!stack.isEmpty() && stack.peek() == popped[j]) {
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }

    // 剑指 Offer 32 - I. 从上到下打印二叉树  medium
    // Queue<TreeNode> q = new LinkedList<>(); offer(), poll(), size(); BFS
    public int[] levelOrder_1(TreeNode root) {
        if (root == null) return new int[0];
        ArrayList<Integer> ans = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            ans.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        int[] res = new int[ans.size()];
        for(int i = 0; i < ans.size(); i++)
            res[i] = ans.get(i);
        return res;
    }

    // 剑指 Offer 32 - II. 从上到下打印二叉树 II  easy
    // List<List<Integer>> res = new ArrayList<>(); BFS
    public List<List<Integer>> levelOrder_2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                assert node != null;
                tmp.add(node.val);
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }

    // 剑指 Offer 32 - III. 从上到下打印二叉树 III  medium  之字形
    // 双端队列，LinkedList, addLast(), addFirst()
    public List<List<Integer>> levelOrder_3(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) queue.offer(root);

        while (!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                assert node != null;

                if (res.size() % 2 == 0) tmp.addLast(node.val);
                else tmp.addFirst(node.val);

                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            res.add(tmp);
        }
        return res;
    }

    // 剑指 Offer 33. 二叉搜索树的后序遍历序列  medium
    // 递归，寻找第一个大于根节点的节点，二叉搜索树：左子树区间所有节点小于根节点，右子树区间所有节点大于根节点
    // 后序遍历中，数组最后一个值为根节点的值
    public boolean verifyPostorder(int[] postorder) {
        return recur(postorder, 0, postorder.length - 1);
    }

    boolean recur(int[] postorder, int i, int j) {
        if (i >= j) return true;
        int p = i;
        while (postorder[p] < postorder[j]) p++;
        int m = p;
        while (postorder[p] > postorder[j]) p++;
        return p == j && recur(postorder, i, m - 1) && recur(postorder, m, j - 1);
    }

    // 剑指 Offer 34. 二叉树中和为某一值的路径  medium
    // dfs, res.add(new ArrayList<>(tmp));, tmp.remove(tmp.size() - 1);
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        dfs(root, res, tmp, 0, target);

        return res;
    }

    private void dfs(TreeNode root, List<List<Integer>> res, List<Integer> tmp, int sum, int target) {
        if (root == null) return;
        else {
            tmp.add(root.val);
            sum += root.val;
            if (sum == target && root.left == null && root.right == null) res.add(new ArrayList<>(tmp));
            dfs(root.left, res, tmp, sum, target);
            dfs(root.right, res, tmp, sum, target);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 剑指 Offer 35. 复杂链表的复制  medium
    // HashMap<Node, Node>, map.put(cur, new Node(cur.val)), map.get(cur).next = map.get(cur.next)
    // 原地解法，复制一个新节点在原节点之后
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while(cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        // 4. 构建新链表的 next 和 random 指向
        while(cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        // 5. 返回新链表的头节点
        return map.get(head);
    }

    // 剑指 Offer 36. 二叉搜索树与双向链表  medium
    // 中序遍历，双向循环链表，dfs
//    Node pre, head;
//    public Node treeToDoublyList(Node root) {
//        if(root == null) return null;
//        dfs(root);
//        head.left = pre;
//        pre.right = head;
//        return head;
//    }
//    void dfs(Node cur) {
//        if(cur == null) return;
//        dfs(cur.left);
//        if(pre != null) pre.right = cur;
//        else head = cur;
//        cur.left = pre;
//        pre = cur;
//        dfs(cur.right);
//    }

    // 剑指 Offer 37. 序列化二叉树  hard
    // BFS, Queue, StringBuilder, str.deleteCharAt(), toString(), str.substring(1, data.length() - 1).split(",")
    // Integer.parseInt(String), String.valueOf(int)
    public String serialize(TreeNode root) {
        if(root == null) return "[]";
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<TreeNode>() {{ add(root); }};
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(node != null) {
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }
            else res.append("null,");
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    public TreeNode deserialize(String data) {
        if(data.equals("[]")) return null;
        String[] vals = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        Queue<TreeNode> queue = new LinkedList<TreeNode>() {{ add(root); }};
        int i = 1;
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(!vals[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if(!vals[i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }

    // 剑指 Offer 38. 字符串的排列  medium
    // dfs, HashSet 去重，boolean[] 标识是否用过，全排列
    // 排序去重 ：if (i > 0 && !vis[i - 1] && cs[i] == cs[i - 1]) continue;
    public String[] permutation(String s) {
        char[] cs = s.toCharArray();
        Set<String> set = new HashSet<>();
        boolean[] vis = new boolean[s.length()];
        dfs(cs, 0, "", set, vis);
        String[] res = new String[set.size()];
        int idx = 0;
        for (String str : set) res[idx++] = str;
        return res;
    }

    void dfs(char[] cs, int n, String cur, Set<String> set, boolean[] vis) {
        if (n == cs.length) {
            set.add(cur);
        } else {
            for (int i = 0; i < cs.length; i++) {
                if (!vis[i]) {
                    vis[i] = true;
                    dfs(cs, n+1, cur + String.valueOf(cs[i]), set, vis);
                    vis[i] = false;
                }
            }
        }
    }

    // 剑指 Offer 39. 数组中出现次数超过一半的数字  easy
    // 摩尔投票法，哈希表统计法，数组排序法
    public int majorityElement(int[] nums) {
        int ans = nums[0], index = 0;
        for (int num : nums) {
            if (num == ans) index++;
            else if (index > 0) index--;
            else {
                ans = num;
                index++;
            }
        }
        return ans;
    }

    // 剑指 Offer 40. 最小的k个数  easy
    // 基于快速排序的数组划分，Arrays.copyOf(arr, k)
    public int[] getLeastNumbers(int[] arr, int k) {
        quickSort(arr, 0, arr.length - 1, k);
        return Arrays.copyOf(arr, k);
    }

    private void quickSort(int[] arr, int l, int r, int k) {
        if (l >= r) return;

        int m = l;
        for (int i = m; i <= r; i++) {
            if (i != m && arr[i] < arr[l]) {
                m++;
                swap(arr, m, i);
            }
        }
        swap(arr, l, m);
        if (m == k) return;
        else if (m > k) quickSort(arr, l, m - 1, k);
        else quickSort(arr, m + 1, arr.length - 1, k);
    }

    private void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    // 大根堆，PriorityQueue
    // Queue<Integer> pq = new PriorityQueue<>((v1, v2) -> v2 - v1);
    public int[] getLeastNumbers_pq(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        // 默认是小根堆，实现大根堆需要重写一下比较器。
        Queue<Integer> pq = new PriorityQueue<>((v1, v2) -> v2 - v1);
        for (int num : arr) {
            if (pq.size() < k) pq.offer(num);
            else if (num < pq.peek()) {
                pq.poll();
                pq.offer(num);
            }
        }

        // 返回堆中的元素
        int[] res = new int[pq.size()];
        int idx = 0;
        for(int num : pq) {
            res[idx++] = num;
        }
        return res;
    }

    // 剑指 Offer 41. 数据流中的中位数  hard
    // 大根堆 + 小根堆，PriorityQueue 默认为小根堆，PriorityQueue<>((x, y) -> (y - x)): 创建大根堆

    // 剑指 Offer 42. 连续子数组的最大和  easy
    // 贪心算法，动态规划，原地
    public int maxSubArray(int[] nums) {
        int ans = nums[0], sum = 0;
        for (int num : nums) {
            sum += num;
            if (sum > ans) ans = sum;
            if (sum < 0) sum = 0;
        }
        return ans;
    }

    // 剑指 Offer 43. 1～n 整数中 1 出现的次数  hard
    // 按位分析，根据当前位的值来分类，int high = n / 10, cur = n % 10, low = 0;
    public int countDigitOne(int n) {
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0) {
            if(cur == 0) res += high * digit;
            else if(cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }

    // 剑指 Offer 44. 数字序列中某一位的数字  medium
    // 数学题，找规律
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit; // 2.
        return Long.toString(num).charAt((n - 1) % digit) - '0'; // 3.
    }

    // 剑指 Offer 45. 把数组排成最小的数  medium
    // 自定义排序，Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
    // Comparator接口可以实现自定义排序，实现Comparator接口时，要重写compare方法：
    // int compare(Object o1, Object o2) 返回一个基本类型的整型
    // 如果要按照升序排序,则o1 小于o2，返回-1（负数），相等返回0，01大于02返回1（正数）
    // 如果要按照降序排序,则o1 小于o2，返回1（正数），相等返回0，01大于02返回-1（负数）
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i = 0; i < nums.length; i++)
            strs[i] = String.valueOf(nums[i]);
        Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder res = new StringBuilder();
        for(String s : strs)
            res.append(s);
        return res.toString();
    }

    // 剑指 Offer 46. 把数字翻译成字符串  medium
    // dp, String.valueOf(), s.substring(findex, lindex), temp.compareTo("10") >= 0 && temp.compareTo("25"
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= s.length(); i ++){
            String temp = s.substring(i-2, i);
            if(temp.compareTo("10") >= 0 && temp.compareTo("25") <= 0)
                dp[i] = dp[i-1] + dp[i-2];
            else
                dp[i] = dp[i-1];
        }
        return dp[s.length()];
    }

    // 剑指 Offer 47. 礼物的最大价值  medium
    // 动态规划，int[][] dp = new int[m+1][n+1], 默认 0 行 0 列均为 0
    public int maxValue(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m+1][n+1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]) + grid[i-1][j-1];
            }
        }
        return dp[m][n];
    }

    // 剑指 Offer 48. 最长不含重复字符的子字符串  medium
    // toCharArray(), HashMap<Character, Boolean>, containsKey()
    // while (str[left] != ch)
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Boolean> map = new HashMap<>();
        int ans = 0, left = 0, right = 0;
        char[] str = s.toCharArray();

        while (right < str.length) {
            char ch = str[right];
            if (!map.containsKey(ch)) {
                map.put(ch, true);
                ans = Math.max(ans, map.size());
                right++;
            } else {
                while (str[left] != ch) {
                    left++;
                    map.remove(str[left]);
                }
                map.remove(str[left]);
                left++;
            }
        }
        return ans;
    }

    // dp, tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
    public int lengthOfLongestSubstring_dp(String s) {
        Map<Character, Integer> dic = new HashMap<>();
        int res = 0, tmp = 0;
        for(int j = 0; j < s.length(); j++) {
            int i = dic.getOrDefault(s.charAt(j), -1); // 获取索引 i
            dic.put(s.charAt(j), j); // 更新哈希表
            tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
            res = Math.max(res, tmp); // max(dp[j - 1], dp[j])
        }
        return res;
    }

    // 剑指 Offer 49. 丑数  medium
    // 最小堆, PriorityQueue(), Set
    // 动态规划，dp[i] = Math.min(dp[a] * 2, Math.min(dp[b] * 3, dp[c] * 5));，if (dp[i] == dp[a] * 2) a++;...
    public int nthUglyNumber(int n) {
        int ans = 1, a = 0, b = 0, c = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = Math.min(dp[a] * 2, Math.min(dp[b] * 3, dp[c] * 5));
            if (dp[i] == dp[a] * 2) a++;
            if (dp[i] == dp[b] * 3) b++;
            if (dp[i] == dp[c] * 5) c++;
        }
        return dp[n - 1];
    }

    // 剑指 Offer 50. 第一个只出现一次的字符  easy
    // HashMap, Integer -> Boolean, LinkedHashMap
    public char firstUniqChar(String s) {
        if (s.length() == 0) return ' ';

        Map<Character, Integer> map = new HashMap<>();
        for (char ch : s.toCharArray()) {
            map.put(ch, map.getOrDefault(ch, 0) + 1);
        }

        for (char ch : s.toCharArray()) {
            if (map.get(ch) == 1) return ch;
        }
        return ' ';
    }

    // 剑指 Offer 51. 数组中的逆序对  hard
    // 归并排序，分而治之，在合并阶段统计逆序对，传递左右边界，计算中值，分别排序，再归并，res += m - i + 1; // 统计逆序对
    // 当 左子数组当前元素 > 右子数组当前元素 时，则 「左子数组当前元素 至 末尾元素」 与 「右子数组当前元素」 构成了若干 「逆序对」
    public int reversePairs(int[] nums) {
        int[] tmp = new int[nums.length];
        return mergeSort(0, nums.length - 1, nums, tmp);
    }

    private int mergeSort(int l, int r, int[] nums, int[] tmp) {
        if (l >= r) return 0;
        int m = (l + r) / 2;
        int res = mergeSort(l, m, nums, tmp) + mergeSort(m + 1, r, nums, tmp);
        // 合并阶段
        int i = l, j = m + 1;
        for (int k = l; k <= r; k++)
            tmp[k] = nums[k];
        for (int k = l; k <= r; k++) {
            if (i == m + 1)
                nums[k] = tmp[j++];
            else if (j == r + 1 || tmp[i] <= tmp[j])
                nums[k] = tmp[i++];
            else {
                nums[k] = tmp[j++];
                res += m - i + 1; // 统计逆序对
            }
        }
        return res;
    }

    // 剑指 Offer 52. 两个链表的第一个公共节点  easy
    // 不能跳过 headB 的首节点 A = A != null ? A.next : headB;
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode dumA = headA, dumB = headB;
        while (headA != headB) {
            if (headA == null) headA = dumB;
            else headA = headA.next;
            if (headB == null) headB = dumA;
            else headB = headB.next;
        }
        return headA;
    }

    // 剑指 Offer 53 - I. 在排序数组中查找数字 I  easy
    // 二分查找，可查找到后中心扩展，亦可分别查找左右边界再相减
    public int search(int[] nums, int target) {
        return search_bin(nums, target, 0, nums.length - 1);
    }

    private int search_bin(int[] nums, int target, int l, int r) {
        if (l > r) return 0;
        int m = (l + r) / 2;
        if (nums[m] == target) {
            int res = 1, left = m - 1, right = m + 1;
            while (right <= r && nums[right] == target) {
                res++;
                right++;
            }
            while (left >= l && nums[left] == target) {
                res++;
                left--;
            }
            return res;
        }
        else if (nums[m] < target) return search_bin(nums, target, m + 1, r);
        else return search_bin(nums, target, l, m - 1);
    }

    public int search_opt(int[] nums, int target) {
        // 搜索右边界 right
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] <= target) i = m + 1;
            else j = m - 1;
        }
        int right = i;
        // 若数组中无 target ，则提前返回
        if(j >= 0 && nums[j] != target) return 0;
        // 搜索左边界 right
        i = 0; j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] < target) i = m + 1;
            else j = m - 1;
        }
        int left = j;
        return right - left - 1;
    }

    // 剑指 Offer 53 - II. 0～n-1中缺失的数字  easy
    // 排序数组中的搜索问题，首先想到 二分法 解决
    public int missingNumber(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = (l + r) / 2;
            if (nums[m] == m) l = m + 1;
            else r = m - 1;
        }
        return l;
    }

    // 剑指 Offer 54. 二叉搜索树的第k大节点  easy
    // 反向中序遍历，右根左，提前返回，if (k == 0) return; if (--k == 0) tmp = root.val;
    int tmp, k;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        kthSearch(root);
        return tmp;
    }

    private void kthSearch(TreeNode root) {
        if (root == null) return;
        kthSearch(root.right);
        if (k == 0) return;
        if (--k == 0) tmp = root.val;
        kthSearch(root.left);
    }

    // 剑指 Offer 55 - I. 二叉树的深度  easy
    // dfs, return Math.max(leftDepth, rightDepth);
    // return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    public int maxDepth(TreeNode root) {
        return dfsMaxDepth(root, 0);
    }

    private int dfsMaxDepth(TreeNode root, int depth) {
        if (root == null) return depth;
        int leftDepth = dfsMaxDepth(root.left, depth + 1);
        int rightDepth = dfsMaxDepth(root.right, depth + 1);
        return Math.max(leftDepth, rightDepth);
    }

    // 剑指 Offer 55 - II. 平衡二叉树  easy
    // 递归，return ans && isBalanced(root.left) && isBalanced(root.right);
    // return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    // 优化思路，后序遍历，自底向上
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        int i = maxDepth(root.left) - maxDepth(root.right);
        boolean ans = false;
        if (i == 0 || i == 1 || i == -1) ans = true;
        return ans && isBalanced(root.left) && isBalanced(root.right);
    }

    // 剑指 Offer 56 - I. 数组中数字出现的次数  medium
    // 位运算，异或，借用辅助值 m 将 nums 分成两个子数组分别异或
    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, n = 0, m = 1;
        for(int num : nums)               // 1. 遍历异或
            n ^= num;
        while((n & m) == 0)               // 2. 循环左移，计算 m
            m <<= 1;
        for(int num: nums) {              // 3. 遍历 nums 分组
            if((num & m) != 0) x ^= num;  // 4. 当 num & m != 0
            else y ^= num;                // 4. 当 num & m == 0
        }
        return new int[] {x, y};          // 5. 返回出现一次的数字
    }

    // 剑指 Offer 56 - II. 数组中数字出现的次数 II  medium
    // 位运算，如果一个数字出现3次，它的二进制每一位也出现的3次
    // >>>表示无符号右移，也叫逻辑右移，即若该数为正，则高位补0，而若该数为负数，则右移后高位同样补0
    // >>表示右移，如果该数为正，则高位补0，若为负数，则高位补1；
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int i = 0; i < 32; i++)
        {
            int oneCount = 0;
            for (int j = 0; j < nums.length; j++)
            {
                oneCount += (nums[j] >>> i) & 1;
            }
            if (oneCount % 3 == 1)
                res |= (1 << i);
        }
        return res;
    }

    // 剑指 Offer 57. 和为s的两个数字  easy
    // 双指针，分别指向 nums 的左右两端
    public int[] twoSum(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] == target) return new int[]{nums[left], nums[right]};
            else if (nums[left] + nums[right] < target) left++;
            else right--;
        }
        return new int[]{0, 0};
    }

    // 剑指 Offer 57 - II. 和为s的连续正数序列  easy
    // 滑动窗口，求和公式，return res.toArray(new int[0][]); toArray(new String[0])
    // new String[0]就是起一个模板的作用，指定了返回数组的类型，0是为了节省空间，因为它只是为了说明返回的类型
    public int[][] findContinuousSequence(int target) {
        ArrayList<int[]> res = new ArrayList<>();
        int sum = 3, left = 1, right = 2;
        while (right <= (target / 2 + 1)) {
            if (sum == target) {
                int[] tmp = new int[right - left +1];
                for (int i = left; i <= right; i++) {
                    tmp[i - left] = i;
                }
                res.add(tmp);
                right++;
                sum += right;
            } else if (sum < target) {
                right++;
                sum += right;
            } else {
                sum -= left;
                left++;
            }
        }
        return res.toArray(new int[0][]);
    }

    // 剑指 Offer 58 - I. 翻转单词顺序  easy
    // str.trim(), str.split(" "), str.substring(begin, last)
    public String reverseWords(String s) {
        if (s.trim().length() == 0) return s.trim();
        String[] words = s.trim().split(" ");
        StringBuilder strs = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            if (words[i].equals("")) continue;
            strs.append(words[i] + " ");
        }
        return strs.toString().substring(0, strs.toString().length() - 1);
    }

    // 剑指 Offer 58 - II. 左旋转字符串  easy
    // return s.substring(n, s.length()) + s.substring(0, n);
    // 利用求余运算，可以简化遍历代码
    public String reverseLeftWords(String s, int n) {
        StringBuilder str = new StringBuilder(s.substring(n));
        str.append(s.substring(0, n));
        return str.toString();
    }

//    public String reverseLeftWords(String s, int n) {
//        StringBuilder res = new StringBuilder();
//        for(int i = n; i < n + s.length(); i++)
//            res.append(s.charAt(i % s.length()));
//        return res.toString();
//    }

    // 剑指 Offer 59 - I. 滑动窗口的最大值  hard
    // 单调队列模板题，Deque<Integer> deque = new LinkedList<>();
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) return new int[0];
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new LinkedList<>();

        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            if (i - deque.peekFirst() == k) {
                deque.pollFirst();
            }
            if (i >= k - 1) {
                res[i - k + 1] = nums[deque.peekFirst()];
            }
        }
        return res;
    }

    // 剑指 Offer 59 - II. 队列的最大值  medium
    // 利用双向队列作为辅助队列，Deque

    // 剑指 Offer 60. n个骰子的点数  medium
    // 动态规划，Arrays.fill(dp, 1.0 / 6.0);
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];
        Arrays.fill(dp, 1.0 / 6.0);
        for (int i = 2; i <= n; i++) {
            double[] tmp = new double[5 * i + 1];
            for (int j = 0; j < dp.length; j++) {
                for (int k = 0; k < 6; k++) {
                    tmp[j + k] += dp[j] / 6.0;
                }
            }
            dp = tmp;
        }
        return dp;
    }

    // 剑指 Offer 61. 扑克牌中的顺子  easy
    // Set, Arrays.sort(), 遍历
    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int king = 0;
        while (nums[king] == 0) {
            king++;
        }

        if (nums[nums.length - 1] - nums[king] > nums.length - 1) return false;
        for (int i = king + 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1]) return false;
        }
        return true;
    }

    // 剑指 Offer 62. 圆圈中最后剩下的数字  easy
    // 约瑟夫环，动态规划，dp[i] = (dp[i − 1] + m) % i
    public int lastRemaining(int n, int m) {
        int x = 0;
        for (int i = 2; i <= n; i++) {
            x = (x + m) % i;
        }
        return x;
    }

    // 剑指 Offer 63. 股票的最大利润  medium
    // 动态规划
    public int maxProfit(int[] prices) {
        if (prices.length == 0) return 0;
        int ans = 0, min = prices[0];
        for (int price : prices) {
            if (price < min) min = price;
            else ans = Math.max(ans, price - min);
        }
        return ans;
    }

    // 剑指 Offer 64. 求1+2+…+n  medium
    // 要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
    // 利用短路与替代 if
    public int sumNums(int n) {
        boolean x = n > 1 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    // 剑指 Offer 65. 不用加减乘除做加法  easy
    // 递归, 逻辑运算
    public int add(int a, int b) {
        if (b == 0) return a;

        // 转换成非进位和 + 进位
        return add(a ^ b, (a & b) << 1);
    }

    // 剑指 Offer 66. 构建乘积数组  medium
    // 本质为两个 dp 数组，分别维护，双向前缀和
    public int[] constructArr(int[] a) {
        if (a.length == 0) return new int[0];
        int[] res = new int[a.length];
        res[0] = 1;
        for (int i = 1; i < res.length; i++) {
            res[i] = res[i - 1] * a[i - 1];
        }
        int tmp = 1;
        for (int j = res.length - 2; j >= 0; j--) {
            tmp *= a[j + 1];
            res[j] *= tmp;
        }
        return res;
    }

    // 剑指 Offer 67. 把字符串转换成整数  medium
    // 考虑边界赋值，toCharArray(), trim(), charAt(), 额外设置符号位
    // if(res > bndry || res == bndry && str.charAt(j) > '7')
    // return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
    public int strToInt(String str) {
        int res = 0, bndry = Integer.MAX_VALUE / 10;
        int i = 0, sign = 1, length = str.length();
        if(length == 0) return 0;
        while(str.charAt(i) == ' ')
            if(++i == length) return 0;
        if(str.charAt(i) == '-') sign = -1;
        if(str.charAt(i) == '-' || str.charAt(i) == '+') i++;
        for(int j = i; j < length; j++) {
            if(str.charAt(j) < '0' || str.charAt(j) > '9') break;
            if(res > bndry || res == bndry && str.charAt(j) > '7')
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            res = res * 10 + (str.charAt(j) - '0');
        }
        return sign * res;
    }

    // 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先  easy
    // 利用好二叉搜索树的性质
    public TreeNode lowestCommonAncestor_1(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return root;

        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor_1(root.left, p, q);
        else if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor_1(root.right, p, q);
        return root;
    }

    // 剑指 Offer 68 - II. 二叉树的最近公共祖先  easy
    // DFS,
    public TreeNode lowestCommonAncestor_2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor_2(root.left, p, q);
        TreeNode right = lowestCommonAncestor_2(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }
}

public class SwordToOffer2 {
    public static void main(String[] args) {
////        int[] arr = {1,2,3};
////        List list = Arrays.asList(1,2,3);
//        SolutionSTO solution = new SolutionSTO();
//        int[] nums1 = new int[]{7,8,10,12,13};
//        int[] nums2 = new int[]{2,4,6,8};
//        System.out.println(solution.isStraight(nums1));
////        System.out.println(-20 >> 2);

        System.out.println();
    }
}
