package com.Lip;

import com.sun.xml.internal.bind.v2.TODO;

import javax.lang.model.element.NestingKind;
import java.lang.reflect.Array;
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
    public int[] findClosedNumbers(int num) {
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

    public int countOneNum(int num) {
        String str = Integer.toBinaryString(num);
        int res = 0;
        for (int i = 0; i < str.length(); i ++) {
            if(str.charAt(i) == '1') {
                res ++;
            }
        }
        return res;
    }

    /*
    面试题 05.06. 整数转换
    异或后求 1 的个数
     */
    public int convertInteger(int A, int B) {
        int temp = (A ^ B);
        return countOneNum(temp);
    }

    /*
    面试题 05.07. 配对交换
    借助掩码 0xaaaaaaaa 和 0x55555555 分别提取偶数位和奇数位。
    偶数位右移，奇数位左移，最后相加就是结果
     */
    public int exchangeBits(int num) {
        int hex_odd = 0xaaaaaaaa, hex_even = 0x55555555;
        return (((num & hex_odd) >> 1) + ((num & hex_even) << 1));
    }

    /*
    面试题 05.08. 绘制直线
     */
    public int[] drawLine(int length, int w, int x1, int x2, int y) {
        int [] screen = new int[length]; int pos = w / 32 * y;
        for(int i = x1; i <= x2; i++) {
            screen[pos + (i / 32)] |= (1 << (31 - (i % 32)));
        }
        return screen;
    }

    /*
    面试题 08.01. 三步问题
    动态规划
     */
    public int waysToStep(int n) {
        if (n == 1 || n == 2) return n;
        long m = 1000000007;
        long[] dp = new long[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 4;

        for (int i = 4; i <= n; i++) {
            dp[i] = dp[i-1] + dp[i-2] + dp[i-3];
            dp[i] %= m;
        }

        return (int) dp[n];
    }

    /*
    面试题 08.02. 迷路的机器人
    dfs, list.add(Arrays.asList(x,y));
     */
    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {
        LinkedList<List<Integer>> list = new LinkedList<>();
        dfs(obstacleGrid, list, 0, 0);
        return list;
    }

    private boolean dfs(int[][] obstacleGrid,LinkedList<List<Integer>> list, int x, int y){
        if (x < 0 || x >= obstacleGrid.length ||
                y < 0 || y >= obstacleGrid[0].length ||
                obstacleGrid[x][y] != 0){
            return false;
        }
        obstacleGrid[x][y] = 1;                //设置为访问过
        list.add(Arrays.asList(x,y));          //添加这个点
        if(x == obstacleGrid.length - 1 && y == obstacleGrid[0].length - 1){
            return true;           //到终点了
        }
        if(dfs(obstacleGrid, list, x+1, y)){   //是否这条路径可以到终点
            return true;
        }
        if(dfs(obstacleGrid, list, x, y+1)){   //是否这条路径可以到终点
            return true;
        }
        list.removeLast();                    //从这个点出发无法到达终点，移除这个点
        return false;
    }

    /*
    面试题 08.03. 魔术索引
    间隔跳跃查找
     */
    public int findMagicIndex(int[] nums) {
        for (int i = 0; i < nums.length;) {
            if (i == nums[i]) return i;
            i = Math.max(nums[i], i + 1);
        }
        return -1;
    }

    /*
    面试题 08.04. 幂集
    回溯，https://leetcode-cn.com/problems/power-set-lcci/solution/hui-su-wei-yun-suan-deng-gong-4chong-fang-shi-jie-/
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();

        dfs(res, temp, nums, 0);

        return res;
    }

    private void dfs(List<List<Integer>> res, List<Integer> temp, int[] nums, int index) {
        if (index == nums.length) {
            res.add(new ArrayList<>(temp));
        } else {
            temp.add(nums[index]);
            dfs(res, temp, nums, index + 1);
            temp.remove(temp.size() - 1);
            dfs(res, temp, nums, index + 1);
        }
    }

    /*
    面试题 08.05. 递归乘法
    位运算
    if ((A & 1) == 1) { // 当B为奇数时，B移位操作导致丢失，需要补偿A
        return multiply(A >> 1, B << 1) + B;
    }
     */
    public int multiply(int A, int B) {
        if (A > B) {
            int temp = A;
            A = B; B = temp;
        }
        if (A == 0) return 0;
        if (A == 1) return B;
//        return (B << 1) + multiply(A - 2, B);
        if ((A & 1) == 1) { // 当B为奇数时，B移位操作导致丢失，需要补偿A
            return multiply(A >> 1, B << 1) + B;
        }

        return multiply(A >> 1, B << 1);
    }

    /*
    面试题 08.06. 汉诺塔问题
    递归
     */
    public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
        move(A.size(), A, B, C);
    }

    void move(int n, List<Integer> A, List<Integer> B, List<Integer> C) {
        if (n == 1) {
            C.add(A.remove(A.size() - 1));
        } else {
            B.add(A.remove(A.size() - 1));
            move(n - 1, A, B, C); // 一处递归即可
            C.add(B.remove(B.size() - 1));
        }
    }

    /*
    面试题 08.07. 无重复字符串的排列组合
    回溯，全排列
     */
    public String[] permutation(String S) {
        List<String> list = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        boolean[] visited = new boolean[S.length()];

        dfs(list, sb, visited, S);
        return list.toArray(new String[0]);
    }

    private void dfs(List<String> list, StringBuilder sb, boolean[] visited, String s) {
        if (sb.length() == s.length()) {
            list.add(sb.toString());
        } else {
            for (int i = 0; i < s.length(); i++) {
                if (!visited[i]) {
                    visited[i] = true;
                    sb.append(s.charAt(i));
                    dfs(list, sb, visited, s);
                    sb.deleteCharAt(sb.length() - 1);
                    visited[i] = false;
                }
            }
        }
    }

    /*
    面试题 08.08. 有重复字符串的排列组合
    HashSet，或者排序剪枝去重
     */
    /*
        char[] chars = S.toCharArray();
        Arrays.sort(chars);

        dfs
            if (visited[i]) continue;
            if (i > 0 && !visited[i - 1] && chars[i - 1] == chars[i]) continue;
     */

    /*
    面试题 08.09. 括号
    dfs, 深度优先遍历
     */
    public List<String> generateParenthesis(int n) {
        List<String> res = new LinkedList<>();

        dfs(res, "", n, 0, 0);
        return res;
    }

    private void dfs(List<String> res, String s, int n, int left, int right) {
        if (right == n) res.add(s);
        else {
            if (left < n) dfs(res, s + "(", n, left + 1, right);
            if (right < left) dfs(res, s + ")", n, left, right + 1);
        }
    }

    /*
    面试题 08.10. 颜色填充
    染色本身可以看做一种标记
    直接 dfs
     */
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (newColor == image[sr][sc]) return image;

        boolean[][] visited = new boolean[image.length][image[0].length];
        pointFill(image, sr, sc, newColor, visited, image[sr][sc]);

        return image;
    }

    private void pointFill(int[][] image, int r, int c, int target, boolean[][] visited, int ori) {
        if (r >= image.length || r < 0 || c >= image[0].length || c < 0 || visited[r][c] || image[r][c] != ori) {
            return;
        }
        visited[r][c] = true;
        image[r][c] = target;
        pointFill(image, r + 1, c, target, visited, ori);
        pointFill(image, r - 1, c, target, visited, ori);
        pointFill(image, r, c + 1, target, visited, ori);
        pointFill(image, r, c - 1, target, visited, ori);
    }

    /*
    面试题 08.11. 硬币
    动态规划，先遍历硬币，保证在考虑一枚硬币的情况时，没有较大的硬币影响
    最终每种组合情况，都是以硬币的面额大小非递减组合。避免了原先调换顺序后重复计算的情况。
     */
    public int waysToChange(int n) {
        long[] dp = new long[n + 1];
        dp[0] = 1;
        int[] coins = new int[]{1,5,10,25};

        for(int coin : coins) {
            for(int i = coin; i <= n; i++) {
                dp[i] = (dp[i] + dp[i - coin]) % 1000000007;
            }
        }

        return (int)dp[n];
    }

    /*
    面试题 08.12. 八皇后
    需要保证皇后不能在同一行， 同一列， 同一条对角线上
    按行搜索，根据不能在同一列和不能在同一对角线的情况下剪枝
     */
    public List<List<String>> solveNQueens(int n) {
        boolean []visit = new boolean[n]; int []list = new int[n];
        List<List<String>> retList= new ArrayList<List<String>>();
        visitMap(visit, n, 0, list, retList);
        return retList;
    }

    private void visitMap(boolean [] visit, int n, int col, int []list, List<List<String>> retList){
        if(col == n){
            List<String> rl = new ArrayList<String>();
            for(int i = 0; i < n; i++){
                StringBuilder s = new StringBuilder();
                for(int j = 0; j < n; j++) s.append(list[i] == j ? "Q" : ".");
                rl.add(s.toString());
            }
            retList.add(rl);
            return;
        }
        for(int i = 0; i < n; i++){
            if(visit[i]) continue;
            if(!verify(list, col, i)) continue;
            visit[i] = true;list[col] = i;
            visitMap(visit, n, col+1, list, retList);
            visit[i] = false;list[col] = 0;
        }
    }

    private boolean verify(int []list, int n, int x){
        for(int i=0;i<n;i++) if(Math.abs(i-n)==Math.abs(list[i]-x)) return false;
        return true;
    }

    /*
    面试题 08.13. 堆箱子
    回溯会超时，采用排序后 dp
    对于dp[i]意思是第i个箱子能达到的最大高度
     */
    public int pileBox(int[][] box) {
        Arrays.sort(box, (a, b) -> {
            if (a[0] != b[0]) {
                return a[0] - b[0];
            } else {
                return b[1] - a[1];
            }
        });
        int m = box.length;
        int max = 0;
        int[] dp = new int[m + 1];
        for (int i = 1; i < m + 1; i++) {
            dp[i] = box[i - 1][2];
            for (int j = 1; j < m + 1; j++) {
                if (box[j - 1][1] < box[i - 1][1] && box[j - 1][2] < box[i - 1][2]) {
                    dp[i] = Math.max(dp[i], dp[j] + box[i - 1][2]);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    /*
    面试题 08.14. 布尔运算
    区间dp
     */
    public int countEval(String s, int result) {
        //特例
        if (s.length() == 0) {
            return 0;
        }
        if (s.length() == 1) {
            return (s.charAt(0) - '0') == result ? 1 : 0;
        }
        char[] ch = s.toCharArray();
        //定义状态
        int[][][] dp = new int[ch.length][ch.length][2];
        //base case
        for (int i = 0; i < ch.length; i++) {
            if (ch[i] == '0' || ch[i] == '1') {
                dp[i][i][ch[i] - '0'] = 1;
            }
        }
        //套区间dp模板
        //枚举区间长度len，跳步为2，一个数字一个符号
        for (int len = 2; len <= ch.length; len += 2) {
            //枚举区间起点，数字位，跳步为2
            for (int i = 0; i <= ch.length - len; i += 2) {
                //区间终点，数字位
                int j = i + len;
                //枚举分割点，三种 '&','|', '^'，跳步为2
                for (int k = i + 1; k <= j - 1; k += 2) {
                    if (ch[k] == '&') {
                        //结果为0 有三种情况： 0 0, 0 1, 1 0
                        //结果为1 有一种情况： 1 1
                        dp[i][j][0] += dp[i][k - 1][0] * dp[k + 1][j][0] + dp[i][k - 1][0] * dp[k + 1][j][1] + dp[i][k - 1][1] * dp[k + 1][j][0];
                        dp[i][j][1] += dp[i][k - 1][1] * dp[k + 1][j][1];
                    }
                    if (ch[k] == '|') {
                        //结果为0 有一种情况： 0 0
                        //结果为1 有三种情况： 0 1, 1 0, 1 1
                        dp[i][j][0] += dp[i][k - 1][0] * dp[k + 1][j][0];
                        dp[i][j][1] += dp[i][k - 1][0] * dp[k + 1][j][1] + dp[i][k - 1][1] * dp[k + 1][j][0] + dp[i][k - 1][1] * dp[k + 1][j][1];
                    }
                    if (ch[k] == '^') {
                        //结果为0 有两种情况： 0 0, 1 1
                        //结果为1 有两种情况： 0 1, 1 0
                        dp[i][j][0] += dp[i][k - 1][0] * dp[k + 1][j][0] + dp[i][k - 1][1] * dp[k + 1][j][1];
                        dp[i][j][1] += dp[i][k - 1][1] * dp[k + 1][j][0] + dp[i][k - 1][0] * dp[k + 1][j][1];
                    }
                }
            }
        }
        return dp[0][ch.length - 1][result];
    }

    /*
    面试题 10.01. 合并排序的数组
    双指针，逆序遍历
     */
    public void merge(int[] A, int m, int[] B, int n) {
        int length = m + n - 1, fir = m - 1, snd = n - 1;

        for (int i = length; i >= 0; i--) {
            if (fir < 0) {
                A[i] = B[snd];
                snd--;
                continue;
            }
            if (snd < 0) {
                break;
            }
            if (A[fir] > B[snd]) {
                A[i] = A[fir];
                fir--;
            } else {
                A[i] = B[snd];
                snd--;
            }
        }
    }

    /*
    面试题 10.02. 变位词组
    对字符串进行排序，并作为 hashMap 的键
    优化思路，利用一个大小为 26 的数组进行计数，然后对计数后的数组统计值进行拼接，作为哈希表的 key，从而实现线性复杂度
     */
    public List<List<String>> groupAnagrams(String[] strs) {

        HashMap<String, List<String>> map = new HashMap<>();

        for (String str : strs) {
            char[] chs = str.toCharArray();
            Arrays.sort(chs);
            if (map.containsKey(Arrays.toString(chs))) {
                map.get(Arrays.toString(chs)).add(str);
            } else {
                map.put(Arrays.toString(chs), new ArrayList<String>(){{add(str);}});
            }
        }

        return new ArrayList<List<String>>(map.values());
    }

    public List<List<String>> groupAnagrams_opt(String[] ss) {
        List<List<String>> ans = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String s : ss) {
            int[] cnts = new int[26];
            for (char c : s.toCharArray()) cnts[c - 'a']++;
            StringBuilder sb = new StringBuilder();
            for (int i : cnts) sb.append(i + "_");
            String key = sb.toString();
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(s);
            map.put(key, list);
        }
        for (String key : map.keySet()) ans.add(map.get(key));
        return ans;
    }

    /*
    面试题 10.03. 搜索旋转数组
    若有多个相同元素，返回索引值最小的一个，即，if(arr[l] == arr[mid]) l++;
    然后根据 mid 的左右端哪段有序进行分类，注意相等的处理，if(arr[l] == arr[mid]) l++;
     */
    public int search(int[] arr, int target) {
        int n = arr.length;
        int result = -1;
        int l = 0, r = n - 1;
        while(l <= r){
            int mid = (l + r + 1) >> 1;
            if(arr[l] == target) return l;
            else if(arr[l] == arr[mid]) l++;
            else if(arr[l] < arr[mid]){
                if(arr[l] > target || arr[mid] < target) l = mid;
                else{
                    l = l + 1;
                    r = mid;
                }
            }else{
                if(arr[l] > target && arr[mid] < target) l = mid;
                else{
                    l = l + 1;
                    r = mid;
                }
            }
        }
        return result;
    }

    /*
    面试题 10.05. 稀疏数组搜索
    当前位置为""，且 mid > left，mid前移
    注意给左移操作符加括号，否则优先级会导致出错
     */
    public int findString(String[] words, String s) {
        int left = 0, right = words.length - 1;

        while (left <= right) {
            int mid = left + ((right - left) >> 1);

            while (mid > left && words[mid].equals("")) {
                mid--;
            }
            if (words[mid].equals(s)) return mid;
            else if (words[mid].compareTo(s) < 0) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    /*
    面试题 10.09. 排序矩阵查找
    二分查找
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length, r = m - 1, c = 0;

        while (r >= 0 && c < n) {
            if (matrix[r][c] == target) return true;
            else if (matrix[r][c] < target) r--;
            else c++;
        }
        return false;
    }

    /*
    面试题 10.10. 数字流的秩
    二分查找，class StreamRank
     */

    /*
    面试题 10.11. 峰与谷
    一次遍历 + 奇偶判断 + 峰谷判断不满足则交换
     */
    public void wiggleSort(int[] nums) {
        if (nums.length != 0) {
            for (int i = 0; i < nums.length; i++) {
                if (i % 2 == 0 && i < nums.length - 1 && nums[i] < nums[i + 1]) {
                    int temp = nums[i];
                    nums[i] = nums[i + 1];
                    nums[i + 1] = temp;
                }
                if (i % 2 == 1 && i < nums.length - 1 && nums[i] > nums[i + 1]) {
                    int temp = nums[i];
                    nums[i] = nums[i + 1];
                    nums[i + 1] = temp;
                }
            }
        }
    }

    /*
    面试题 16.01. 交换数字
    加减或者异或，加减需要考虑溢出问题，所以异或更好
     */
    public int[] swapNumbers(int[] numbers) {
        numbers[0] = numbers[0] ^ numbers[1];
        numbers[1] = numbers[0] ^ numbers[1];
        numbers[0] = numbers[0] ^ numbers[1];
        return numbers;
    }

    /*
    面试题 16.02. 单词频率
    Hashmap，getOrDefault
     */

    /*
    面试题 16.03. 交点
    纯算法题，参数方程
     */

    /*
    面试题 16.04. 井字游戏
    分别检查横和，纵和以及对角线和
     */
    public String tictactoe(String[] board) {
        int length = board.length;
        int heng = 0; //横的和
        int zong = 0; //纵的和
        int left = 0; //左斜线
        int right = 0; //右斜线
        boolean flag = false; //记录有没有空格

        for (int i = 0; i < length; i++) {
            heng = 0; zong = 0;
            for (int j = 0; j < length; j++) {
                heng = heng +  (int) board[i].charAt(j);
                zong = zong + (int) board[j].charAt(i);

                if(board[i].charAt(j) == ' ') flag = true;

            }
            //横纵检查
            if (heng == (int)'X' * length || zong == (int)'X' * length) return "X";
            if (heng == (int)'O' * length || zong == (int)'O' * length) return "O";
            //两条斜线上的相加
            left = left + (int)board[i].charAt(i);
            right = right + (int)board[i].charAt(length - i - 1);
        }
        //两条斜线检查
        if (left == (int)'X' * length || right == (int)'X' * length) return "X";
        if (left == (int)'O' * length || right == (int)'O' * length) return "O";
        if (flag) return "Pending";
        return "Draw";
    }

    /*
    面试题 16.05. 阶乘尾数
    统计有多少个 0, 实际上是统计 2 和 5 一起出现多少对, 不过因为 2 出现的次数一定大于 5 出现的次数
    因此我们只需要检查 5 出现的次数就好了
     */
    public int trailingZeroes(int n) {
        int count = 0;
        while(n >= 5){
            n /= 5;
            count += n;
        }
        return count;
    }

    /*
    面试题 16.06. 最小差
    排序 + 双指针，注意 int 溢出，利用 (long) 进行强制类型转换
     */
    public int smallestDifference(int[] a, int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        long ans = Integer.MAX_VALUE;
        int l = 0, r = 0;

        while (l < a.length || r < b.length) {
            if (l == a.length) {
                ans = Math.min(ans, Math.abs((long)a[l - 1] - (long)b[r++]));
            } else if (r == b.length) {
                ans = Math.min(ans, Math.abs((long)a[l++] - (long)b[r - 1]));
            } else {
                ans = Math.min(ans, Math.abs((long)a[l] - (long)b[r]));
                if (a[l] > b[r]) r++;
                else l++;
            }
        }
        return (int) ans;
    }

    /*
    面试题 16.07. 最大数值
    首先 a - b 得到差值x, 由于是long型，右移63位得到符号位，注意负号不变，那么正数右移63位就是0，负数右移63位就是-1
    那么得出我们的计算公式 (1 + k) * a - b * k
     */
    public int maximum(int a, int b) {
        long x = (long) a - (long) b;
        int k = (int) (x >> 63);

        return (1 + k) * a - b * k;
    }

    /*
    面试题 16.08. 整数的英语表示
    递归调用
     */
    private final int[] NUMBER = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 1000, 1000000, 1000000000};
    private final String[] WORDS = {"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen",
            "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety", "Hundred", "Thousand",
            "Million", "Billion"};

    public String numberToWords(int num) {
        // 后面可能对 NUMBER[i] 求余
        if (num == 0) return "Zero";
        // 从 NEMBERS 的最大索引处开始
        int i = 30;
        // 找到num最大的数位
        while (i >= 0 && NUMBER[i] > num) --i;
        StringBuilder res = new StringBuilder();
        // 90以前的 NUMBER 转WORDS不需要前缀
        if (NUMBER[i] <= 90)
            res.append(WORDS[i]);
        else
            // e.g. 100 = "One" + " " + "Hundred", 90以后的NUMBER转WORDS需要一个前缀
            res.append(numberToWords(num / NUMBER[i])).append(" ").append(WORDS[i]);
        // 下一位
        if (num % NUMBER[i] > 0)
            res.append(" ").append(numberToWords(num % NUMBER[i]));
        return res.toString();
    }

    /*
    面试题 16.09. 运算
    快速幂，提前准备好二进制的正值和负值
     */

    /*
    面试题 16.10. 生存人数
    利用数组记录出生时刻, 死亡时刻
    再根据前缀和, 求最大值, 并标记年份
     */
    public int maxAliveYear(int[] birth, int[] death) {
        // 先统计每年的人口数变化
        int[] change = new int[102];
        for (int i = 0; i < birth.length; i++) {
            // eg:1900年出生的人导致1900年变化人数加1，存储在change[0]
            change[birth[i] - 1900]++;
            // eg:1900年死亡的人导致1901年变化人数减1，存储在change[1]
            change[death[i] - 1899]--;
        }
        int maxAlive = 0;
        int curAlive = 0;
        int theYear = 1900;
        // 再根据每年变化人数求一个最大值
        for (int i = 0; i < 101; i++) {
            curAlive += change[i];
            if (curAlive > maxAlive) {
                maxAlive = curAlive;
                theYear = 1900 + i;
            }
        }
        return theYear;
    }

    /*
    面试题 16.11. 跳水板
    数学解法，考虑边界
     */
    public int[] divingBoard(int shorter, int longer, int k) {
        if (k == 0) return new int[0];
        if (shorter == longer) return new int[]{shorter * k};

        int[] ans = new int[k + 1];
        for (int i = 0; i <= k; i++) {
            ans[i] = shorter * (k - i) + longer * i;
        }

        return ans;
    }

    /*
    面试题 16.13. 平分正方形
    求过两个正方形中心的直线即可，如果两者中心重合，取垂直于x轴的直线。再根据斜率判断与正方形哪条边相交。
     */
    public double[] cutSquares(int[] square1, int[] square2) {
        //第一个正方形的中心点，x,y坐标及正方形边长
        double x1 = square1[0] + square1[2]/2.0, y1 = square1[1] + square1[2]/2.0;
        int d1 = square1[2];
        //第二个正方形的中心点，x,y坐标及正方形边长
        double x2 = square2[0] + square2[2]/2.0, y2 = square2[1] + square2[2]/2.0;
        int d2 = square2[2];
        //结果集
        double[] res = new double[4];
        //两个中心坐标在同一条x轴上，此时两条直线的斜率都是无穷大
        if(x1 == x2){
            res[0] = x1;
            res[1] = Math.min(square1[1], square2[1]);
            res[2] = x2;
            res[3] = Math.max(square1[1] + d1, square2[1] + d2);
        }else{
            //斜率存在，则计算斜率
            double k = (y1 - y2)/(x1 - x2);
            double b = y1 - k*x1;
            //斜率绝对值大于1，说明与正方形的上边和下边相交
            if(Math.abs(k) > 1){
                res[1] = Math.min(square1[1],square2[1]);
                res[0] = (res[1] - b)/k;
                res[3] = Math.max(square1[1] + d1,square2[1] + d2);
                res[2] = (res[3] - b)/k;
            }else{
                //斜率绝对值小于等于1，说明与正方形的左边和右边相交
                res[0] = Math.min(square1[0],square2[0]);
                res[1] = res[0]*k + b;
                res[2] = Math.max(square1[0] + d1,square2[0] + d2);
                res[3] = res[2]*k + b;
            }
        }
        // 题目要求x1 < x2,如果结果不满足，我们交换两个点的坐标即可
        if(res[0] > res[2]){
            swap(res, 0 ,2);
            swap(res, 1, 3);
        }
        return res;
    }

    public void swap(double[] res, int x, int y){
        double temp = res[x];
        res[x] = res[y];
        res[y] = temp;
    }

    /*
    面试题 16.14. 最佳直线
    动态规划
    i从后往前遍历 dp[i][k]表示以i为开始节点[i,n)范围内斜率为k的最大数量
    j 从后往前遍历 ,i j两点的斜率为k，则动态转移方程为dp[i][k] = dp[j][k] + 1，因为斜率相等则共线
    因为斜率k是double类型，有无限多个，所以dp不能用二维数组表示，要用一维数组+哈希表
    Double.NEGATIVE_INFINITY 负无穷 Double.POSITIVE_INFINITY 正无穷 Double.NaN 非数
     */
    public int[] bestLine(int[][] points) {
        int n = points.length;
        List<Map<Double, Integer>> dp = new ArrayList<>(n);
        for (int i = 0; i < n; i++) dp.add(new HashMap<>());
        int maxCnt = 0, l = 0, r = 0;
        for (int i = n - 2; i >= 0; i--) {
            for (int j = n - 1; j > i; j--) {
                int dx = points[i][0] - points[j][0], dy = points[i][1] - points[j][1];
                double k = dx == 0 ? Double.POSITIVE_INFINITY : dy == 0 ? 0 : (double) dy / dx;
                int cnt = dp.get(j).getOrDefault(k, 1) + 1;
                dp.get(i).put(k, cnt);
                if (maxCnt > cnt) continue;
                maxCnt = cnt;
                l = i;
                r = j;
            }
        }
        return new int[]{l, r};
    }

    /*
    面试题 16.15. 珠玑妙算
    设置一个长26的数组map（目的是将RYGB对应到数组的index中）
    遍历，如果solution和guess对应元素相等，则直接real++
    若不相等，判断map中sol元素是否小于0（代表之前存过guess的元素），存在则fake++，然后更新map[sol - 'A']++;
    对map中的guess元素做同等判断
     */
    public int[] masterMind(String solution, String guess) {
        int fake = 0, real = 0;
        int[] map = new int[26];

        for(int i = 0; i < 4; i++){
            char sol = solution.charAt(i), gue = guess.charAt(i);

            if(sol == gue) real++;
            else{
                if(map[sol - 'A'] < 0) fake++;
                map[sol - 'A']++;

                if(map[gue - 'A'] > 0) fake++;
                map[gue - 'A']--;
            }
        }

        int[] ans = {real, fake};
        return ans;
    }


}


public class LeetCode_Mst {
    public static void main(String[] args) {
//        int[] arr = {1,2,3};
//        List list = Arrays.asList(1,2,3);
        Solution_Mst solution = new Solution_Mst();
        int[] nums1 = new int[]{15,16,19,20,25,1,3,4,5,7,10,14};
        int[] nums2 = new int[]{1,2,2,2,1};
//        ListNode l1 = new ListNode(1);
//        l1.next = new ListNode(2);
//        l1.next.next = new ListNode(1);
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(2);
        TreeNode root2 = new TreeNode(2);
        System.out.print(solution.search(nums1, 5));
    }
}
