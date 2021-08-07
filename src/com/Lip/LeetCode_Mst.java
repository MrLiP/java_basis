package com.Lip;

import java.util.*;

class Solution_Mst {

    // 面试题 01.01. 判定字符是否唯一
    // HashMap, int[26] 统计所有的字母
    // 基于位运算的方法，使用一个int类型的变量（下文用mark表示）来代替长度为26的bool数组
    public boolean isUnique(String astr) {
        int[] chs = new int[26];

        for(int i = 0; i < astr.length(); i++) {
            char ch = astr.charAt(i);
            chs[ch - 'a'] += 1;
        }

        for (int i : chs) {
            if (i > 1) return false;
        }
        return true;
    }

    // 面试题 01.02. 判定是否互为字符重排
    // 1.先排序再比较是否相等，s1.toCharArray(), Arrays.sort(), new String(...).equals(new String(...))
    // 2. HashMap 计数
    // 3. 桶计数，即用数组来统计每个字符次数，本质类似方法二
    public boolean CheckPermutation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;

        int[] chs = new int[26];

        for (int i = 0; i < s1.length(); i++) {
            chs[s1.charAt(i) - 'a'] += 1;
            chs[s2.charAt(i) - 'a'] -= 1;
        }

        for (int i : chs) {
            if (i != 0) return false;
        }
        return true;
    }

    // 面试题 01.03. URL化
    // 倒序计算，从后往前为数组赋值
    // StringBuilder
    // replace(), substring()
    public String replaceSpaces(String S, int length) {
        char[] chars = S.toCharArray();
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < length; i++) {
            char ch = S.charAt(i);
            if (ch == ' ') {
                str.append("%20");
            } else str.append(ch);
        }
        return str.toString();
    }

    // 面试题 01.04. 回文排列
    // set 去重，偶数情况直接去除，最后判断 set 的 size()
    // 位运算，用两个 long 判断 128 位的字符，一个 long 位数为 64位 (8字节)，int 为 32位
    public boolean canPermutePalindrome(String s) {
        int[] chs = new int[128];

        for (char ch : s.toCharArray()) {
            chs[ch] += 1;
        }

        boolean flag = false;
        for (int i : chs) {
            if (i % 2 == 1) {
                if (!flag) flag = true;
                else return false;
            }
        }
        return true;
    }

    public boolean canPermutePalindrome_opt(String s) {
        int[] map = new int[128];
        int count = 0;
        for (char ch : s.toCharArray()) {
            //如果当前位置的字符数量是奇数，再加上当前字符
            //正好是偶数，表示消掉一个，我们就把count减一，
            //否则count就加一
            if ((map[ch]++ & 1) == 1) {
                count--;
            } else {
                count++;
            }
        }
        return count <= 1;
    }

    // 面试题 01.05. 一次编辑
    // 动态规划，双指针模拟
    public boolean oneEditAway(String first, String second) {
        if (Math.abs(first.length() - second.length()) > 1) return false;

        int[][] dp = new int[first.length() + 1][second.length() + 1];

        for (int i = 1; i < first.length(); i++) dp[0][i] = i - 1;
        for (int j = 1; j < second.length(); j++) dp[j][0] = j - 1;

        for (int i = 1; i <= first.length(); i++) {
            for (int j = 1; j <= second.length(); j++) {
                if (first.charAt(i - 1) == second.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = dp[i - 1][j - 1] + 1;
                dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
                dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
            }
        }

        return dp[first.length()][second.length()] <= 1;
    }

    // 面试题 01.06. 字符串压缩
    // StringBuilder, 比较前后是否一致
    public String compressString(String S) {
        StringBuilder str = new StringBuilder();

        for (int i = 0; i < S.length(); i++) {
            int n = 1;
            while (i + 1 < S.length() && S.charAt(i) == S.charAt(i + 1)) {
                n += 1;
                i += 1;
            }
            str.append(S.charAt(i));
            str.append(n);
        }

        return S.length() <= str.length() ? S : str.toString();
    }

    // 面试题 01.07. 旋转矩阵
    // 原地旋转，for (int i = 0; i < n/2; i++)，for (int j = 0; j < (n+1)/2; j++)
    // matrix[i][j] << matrix[n - j - 1][i] << matrix[n - i - 1][n - j - 1] << matrix[j][n - i - 1]
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int mid = n / 2;

        for (int i = 0; i < n/2; i++) {
            for (int j = 0; j < (n+1)/2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }

    // 面试题 01.08. 零矩阵
    // HashSet, 遍历两次，第一次保存 0 的行列值，第二次将对应行列置 0
    // HashSet >> boolean[]，优化思路
    // 优化思路，利用矩阵的第一行和第一列来代替之前的标记数组
    public void setZeroes(int[][] matrix) {
        HashSet<Integer> row = new HashSet<>();
        HashSet<Integer> col = new HashSet<>();

        int m = matrix.length, n = matrix[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row.add(i);
                    col.add(j);
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row.contains(i) || col.contains(j)) matrix[i][j] = 0;
            }
        }
    }

    // 面试题 01.09. 字符串轮转
    // 先判断长度是否相同，不相同返回false，其次拼接两个s2，则如果是由s1旋转而成，则拼接后的s一定包含s1
    // String s = s1 + s2;  return s.contains(s1);
    public boolean isFlipedString(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        if (s1.equals(s2) || s1.length() == 0) return true;

        for (int i = 0; i < s1.length(); i++) {
            int j = 0;
            if (s2.charAt(i) == s1.charAt(j)) {
                while (j < s2.length() && s2.charAt((i + j) % s2.length()) == s1.charAt(j)) {
                    j++;
                }
                if (j == s2.length()) return true;
            }
        }

        return false;
    }

    // 面试题 02.01. 移除重复节点 >> 链表
    // HashSet, 存储所有出现过的节点，ListNode dummy, ListNode pre
    public ListNode removeDuplicateNodes(ListNode head) {
        HashSet<Integer> set = new HashSet<>();
        ListNode dummy = new ListNode(-1);
        ListNode pre = dummy;
        dummy.next = head;

        while (head != null) {
            if (set.contains(head.val)) {
                head = head.next;
                pre.next = head;
            } else {
                set.add(head.val);
                pre = head;
                head = head.next;
            }
        }
        return dummy.next;
    }

    // 面试题 02.02. 返回倒数第 k 个节点 >> 链表
    // 快慢指针，slow = k-- <= 0 ? slow.next : slow;
    public int kthToLast(ListNode head, int k) {
        ListNode fast = head;
        for (int i = 0; i < k; i ++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            head = head.next;
        }
        return head.val;
    }

    // 面试题 02.03. 删除中间节点 >> 链表
    // 将自己变为下一节点，再将下一节点剔除
    public void deleteNode(ListNode node) {
        node.val=node.next.val;
        node.next=node.next.next;
    }

    // 面试题 02.04. 分割链表 >> 链表
    // 快慢指针，直接在当前列表上操作
    // 或维护两个链表，small 链表按顺序存储所有小于 xx 的节点，large 链表按顺序存储所有大于等于 x 的节点，最后拼接
    public ListNode partition(ListNode head, int x) {
        ListNode dum = new ListNode(-1);
        dum.next = head;
        ListNode fast = dum, slow = dum;

        while (fast.next != null) {
            if (fast.next.val >= x) {
                fast = fast.next;
            } else {
                if (slow == fast) {
                    slow = slow.next;
                    fast = fast.next;
                } else {
                    ListNode cur = fast.next;
                    fast.next = fast.next.next;
                    cur.next = slow.next;
                    slow.next = cur;
                    slow = slow.next;
                }
            }
        }

        return dum.next;
    }

    // 面试题 02.05. 链表求和 >> 链表
    // 根据位次依次求和，设置进位符号 flag，分多种情况讨论
    // 优化，while 循环条件 while(l1 != null || l2 != null || x != 0), 然后依据是否为 null 分开讨论，只关注当前位的和
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        ListNode dum = new ListNode(-1);
        ListNode cur = dum;
        int flag = 0;
        while (l1 != null && l2 != null) {
            int temp = l1.val + l2.val + flag;
            flag = temp / 10;
            cur.next = new ListNode(temp % 10);
            cur = cur.next;
            l1 = l1.next;
            l2 = l2.next;
        }

        if (l1 == null) l1 = l2;
        if (flag == 0) {
            cur.next = l1;
        } else {
            while (flag != 0 && l1 != null) {
                if (l1.val == 9) {
                    cur.next = new ListNode(0);
                } else {
                    cur.next = new ListNode(l1.val + 1);
                    flag = 0;
                }
                cur = cur.next;
                l1 = l1.next;
            }
            if (flag == 1) cur.next = new ListNode(1);
            else cur.next = l1;
        }
        return dum.next;
    }

    public ListNode addTwoNumbers_opt(ListNode l1, ListNode l2) {
        int x = 0;  // 进位
        ListNode dummy = new ListNode(0);   // 哑节点
        ListNode node = dummy;

        while(l1 != null || l2 != null || x != 0) {
            int sum = x;    // 当前位的和
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            node.next = new ListNode(sum % 10);
            x = sum / 10;
            node = node.next;
        }
        return dummy.next;
    }

    // 面试题 02.06. 回文链表 >> 链表
    // 快慢指针找前半部分链表的尾结点
    // 反转后半部分链表
    // 判断是否回文，（恢复链表）
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }

        // 找到前半部分链表的尾节点并反转后半部分链表
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // 判断是否回文
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) {
                result = false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }

        // 还原链表并返回结果
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    private ListNode endOfFirstHalf(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    // 面试题 02.07. 链表相交 >> 链表
    // 只要以相同的速度前进，就一定有机会遇见你
    // 无交点时，会一直执行到两个链表的末尾，curA, curB 都为null,也会跳出循环
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA, curB = headB;

        while (curA != curB) {
            if (curA == null) curA = headB;
            else curA = curA.next;
            if (curB == null) curB = headA;
            else curB = curB.next;
        }

        return curA;
    }

    // 面试题 02.08. 环路检测 >> 链表
    // HashSet, 快慢指针
    // a=c+(n-1)(b+c)a=c+(n−1)(b+c)
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> set = new HashSet<>();

        while (head != null) {
            if (set.contains(head)) return head;
            set.add(head);
            head = head.next;
        }
        return null;
    }

    public ListNode detectCycle_opt(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head;
        while (fast != null) {
            slow = slow.next;
            if (fast.next != null) {
                fast = fast.next.next;
            } else {
                return null;
            }
            if (fast == slow) {
                ListNode ptr = head;
                while (ptr != slow) {
                    ptr = ptr.next;
                    slow = slow.next;
                }
                return ptr;
            }
        }
        return null;
    }

    // 面试题 03.01. 三合一
    // 三个一位数组，或二维数组

    // 面试题 03.02. 栈的最小值 >> 栈
    // 利用辅助栈，或者新建节点类

    // 面试题 03.04. 化栈为队 >> 栈
    // 栈的特性是 FILO（先进后出），队列的特性是 FIFO（先进先出）
    // 用两个栈来模拟队列的特性，一个栈为入队栈，一个栈为出对栈

    // 面试题 03.05. 栈排序 >> 栈
    // 辅助栈，每次插入一个新元素，把它放到它应该放到的位置
    // List.sort((o1, o2) -> (o2 - o1))

    // 面试题 03.06. 动物收容所 >> 队列
    // 双队列，用 int 存储年龄
    // 编号就代表年龄，所以不需要额外存储年龄。。。

    // 面试题 04.01. 节点间通路 >> 图
    // 首先设置访问状态数组 visited，使用DFS「深度优先搜索」进行递归搜索，逐渐压缩搜索区间
    // visited 目的是为了防止出现闭环
    // 采用DFS解决的过程中，应当注意当正序递归搜索时，会出现超时的情况，所以采用逆序搜索的方法
    // if (graph[i][1] == target && helper(graph, start, graph[i][0], visited)) return true;
    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        boolean[] visited = new boolean[graph.length];
        return helper(graph, start, target, visited);
    }

    private boolean helper(int[][] graph, int start, int target, boolean[] visited) {
        for (int i = 0; i < graph.length; i++) {
            if (!visited[i]) {
                if (graph[i][0] == start && graph[i][1] == target) return true;
                visited[i] = true;
                if (graph[i][1] == target && helper(graph, start, graph[i][0], visited)) return true;
                visited[i] = false;
            }
        }
        return false;
    }

    // 面试题 04.02. 最小高度树 >> 二叉树 >> 递归
    // 使用递归的方式，每次取数组中间的值作为当前节点
    public TreeNode sortedArrayToBST(int[] num) {
        if (num.length == 0)
            return null;
        return sortedArrayToBST(num, 0, num.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] num, int start, int end) {
        if (start > end)
            return null;
        int mid = (start + end) >> 1;
        TreeNode root = new TreeNode(num[mid]);
        root.left = sortedArrayToBST(num, start, mid - 1);
        root.right = sortedArrayToBST(num, mid + 1, end);
        return root;
    }

    // 面试题 04.03. 特定深度节点链表
    // 二叉树 BFS + 创建链表
    public ListNode[] listOfDepth(TreeNode tree) {
        List<ListNode> res = new ArrayList<>();
        Queue<TreeNode> list = new LinkedList<>();
        list.add(tree);

        bfs(res, list);
        return res.toArray(new ListNode[0]);
    }

    private void bfs(List<ListNode> res, Queue<TreeNode> list) {
        while (list.size() != 0) {
            ListNode dummy = new ListNode(-1), cur = dummy;
            Queue<TreeNode> temp = new LinkedList<>();
            while (list.size() != 0) {
                TreeNode node = list.poll();
                if (node.left != null) temp.offer(node.left);
                if (node.right != null) temp.offer(node.right);
                cur.next = new ListNode(node.val);
                cur = cur.next;
            }
            cur.next = null;
            res.add(dummy.next);
            list = temp;
        }
    }

    // 面试题 04.04. 检查平衡性
    // 自顶向下递归，利用求左右子树深度的方式确定该节点是否为平衡节点, O(n²)
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        boolean flag = Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1;

        return flag && isBalanced(root.left) && isBalanced(root.right);
    }

    private int maxDepth(TreeNode root) {
        if (root == null) return 1;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    // 后序遍历优化思路，自底向上递归，O(n)
    /*
          class Solution {
              boolean flag = true;
              public boolean isBalanced(TreeNode root) {
                  dfs(root);
                  return flag;
              }
              public int dfs(TreeNode root){
                  if(root == null) return 0;
                  int left = dfs(root.left);
                  int right = dfs(root.right);
                  if(Math.abs(left-right) > 1){
                      flag = false;
                  }
                  return Math.max(left,right) + 1;
              }
          }
     */

    // 面试题 04.05. 合法二叉搜索树
    // 中序遍历，遍历过程中判断当前节点的值是否大于前一个中序遍历到的节点
    public boolean isValidBST(TreeNode root) {
        Deque<TreeNode> stack = new LinkedList<>();
        double inorder = -Double.MAX_VALUE;

        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
            if (root.val <= inorder) {
                return false;
            }
            inorder = root.val;
            root = root.right;
        }
        return true;
    }

    // 面试题 04.06. 后继者
    // 非递归中序遍历，利用一个 flag 记录是否找到了目标 p
    // 一直往左找，并依次加入栈，直到为空。root=root.left
    // 出栈，并判断tag是否为true，如果为true，代表已经找到了p的下一个，直接返回；否则继续判断是否为目标p，如果为p,将tag置为true。
    // 遍历出栈元素的右节点。root=root.right
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        Deque<TreeNode> stack = new LinkedList<>();
        boolean flag = false;

        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (flag) return root;
            if (root == p) flag = true;
            root = root.right;
        }
        return null;
    }

    // 面试题 04.08. 首个共同祖先
    // 递归
    // if (root == null || root == p || root == q) return root;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }

    /*
    面试题 04.09. 二叉搜索树序列
    每一个节点都必须排在它的子孙结点前面
    回溯法 + bfs
     */
    public List<List<Integer>> BSTSequences(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();

        if (root == null) {
            ans.add(path);
            return ans;
        }
        List<TreeNode> queue = new ArrayList<>();
        queue.add(root);
        bfs(ans, path, queue);
        return ans;
    }

    private void bfs(List<List<Integer>> ans, List<Integer> path, List<TreeNode> queue) {
        if (queue.isEmpty()) {
            ans.add(new ArrayList<>(path));
            return;
        }
        List<TreeNode> copy = new ArrayList<>(queue);
        for (int i = 0; i < queue.size(); i++) {
            TreeNode cur = queue.get(i);
            path.add(cur.val);
            queue.remove(i);
            if (cur.left != null) queue.add(cur.left);
            if (cur.right != null) queue.add(cur.right);
            bfs(ans, path, queue);
            path.remove(path.size() - 1);
            queue = new ArrayList<>(copy);
        }
    }

    /*
    面试题 04.10. 检查子树
    递归，从t1树的每个节点开始比对是否能够找到完整的t2树
    先序序列化可以将一棵树的结构和值信息完全保存，所以可以将t1和t2都进行先序序列化，
    如果t2的序列化结果是t1序列化结果的子数组，那么就说明t2是t1的一个子树
     */
    public boolean checkSubTree(TreeNode t1, TreeNode t2) {
        if (t2 == null) return true;
        if (t1 == null) return false;
        return check(t1, t2) || checkSubTree(t1.left, t2) || checkSubTree(t1.right, t2);
    }

    private boolean check(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        else if (t1 == null || t2 == null) return false;
        return t1.val == t2.val && check(t1.left, t2.left) && check(t1.right, t2.right);
    }

    /*
    面试题 04.12. 求和路径
    递归 + dfs
    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        return pathSum(root.left, sum) + pathSum(root.right, sum) + helper(root, sum);
    }
    private int helper(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        int ret = sum == root.val ? 1 : 0;
        sum -= root.val;
        return helper(root.left, sum) + helper(root.right, sum) + ret;
    }
     */
    public int pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();

        dfs(res, path, root, sum);
        return res.size();
    }

    private void dfs(List<List<Integer>> res, List<Integer> path, TreeNode root, int sum) {
        if (root != null) {
            if (path.size() == 0) {
                dfs(res, path, root.left, sum);
                dfs(res, path, root.right, sum);
            }
            path.add(root.val);
            if (sum - root.val == 0) {
                res.add(new ArrayList<>(path));
            }
            dfs(res, path, root.left, sum - root.val);
            dfs(res, path, root.right, sum - root.val);
            path.remove(path.size() - 1);
        }
    }

    /*
    面试题 05.01. 插入
    位运算
    首先弄左右两个遮罩 left, right，然后用这个遮罩去把要保留的区域保存下来，其他位置零
    注意，当 j 为 31 时，不能直接左移 32 位，因为这样移动得到的结果全部是 1，只能让它全部位为 0 才行
     */
    public int insertBits(int N, int M, int i, int j) {
        // 先把相应的位置零
        int allOnes = ~0;
        int left = j == 31 ? 0 : (allOnes << (j + 1));
        int right = (1 << i) - 1;
        int mask = (left | right);
        N &= mask;
        M <<= i;
        return N | M;
    }

    /*
    面试题 05.02. 二进制数转字符串
    小数的二进制是小数部分一直*2取个位数
    StringBuilder
     */
    public String printBin(double num) {
        StringBuilder sb = new StringBuilder("0.");

        while (sb.length() < 32) {
            num *= 2;
            if (num >= 1) {
                sb.append("1");
                num -= 1;
            }
            else sb.append("0");
            if (num == 0) return sb.toString();
        }
        return "ERROR";
    }

    /*
    面试题 05.03. 翻转数位
    动态规划，
    cur：当前位置为止连续1的个数，遇到0归零，遇到1加1
    insert：在当前位置变成1，往前数连续1的最大个数，遇到0变为cur+1，遇到1加1
    res:保存insert的最大值即可


     */
    public int reverseBits(int num) {
        int cur = 0, temp = 0, res = 1;

        for (int i = 0; i < 32; i++) {
            if ((num & 1) == 1) {
                cur += 1;
                temp += 1;
            } else {
                temp = cur + 1;
                cur = 0;
            }
            res = Math.max(res, temp);
            num >>= 1;
        }
        return res;
    }

    /*
    面试题 05.04. 下一个数
    暴力法，位运算
     */
    public static int[] findClosedNumbers(int num) {
        int[] res = {-1, -1};
        long max = (long) num << 1;
        int min = num >> 1;
        int oneCount = countOneNum(num);
        for (int i = num + 1; i <= max; i ++) {
            if(oneCount == countOneNum(i)) {
                res[0] = i;
                break;
            }
        }
        for (int i = num - 1; i >= min; i --) {
            if(oneCount == countOneNum(i)) {
                res[1] = i;
                break;
            }
        }
        return res;
    }

    public static int countOneNum(long num) {
        String str = Long.toBinaryString(num);
        int res = 0;
        for (int i = 0; i < str.length(); i ++) {
            if(str.charAt(i) == '1') {
                res ++;
            }
        }
        return res;
    }


}


public class LeetCode_Mst {
    public static void main(String[] args) {
//        int[] arr = {1,2,3};
//        List list = Arrays.asList(1,2,3);
        Solution_Mst solution = new Solution_Mst();
        int[] nums1 = new int[]{3,2,3,6,4,1,2,3,2,1,2,2};
        int[] nums2 = new int[]{1,2,2,2,1};
//        ListNode l1 = new ListNode(1);
//        l1.next = new ListNode(2);
//        l1.next.next = new ListNode(1);
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(2);
        TreeNode root2 = new TreeNode(2);
        System.out.print(solution.printBin(0.625));
    }
}
