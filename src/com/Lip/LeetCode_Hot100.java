package com.Lip;

import java.util.*;


class Solution {
    // 1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashtable = new HashMap<>();

        for (int i=0;i<nums.length;++i) {
            if (hashtable.containsKey(target-nums[i])) {
                return new int[]{hashtable.get(target-nums[i]), i};
            }
            hashtable.put(nums[i], i);
        }
        return new int[0];
    }

    // 2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            tail.next = new ListNode(carry);
        }
        return head;
    }

    // 3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        int rk = -1, ans = 0;

        for (int i=0;i<n;i++) {
            if (i != 0) {
                occ.remove(s.charAt(i-1));
            }
            while (rk+1 < n && !occ.contains(s.charAt(rk+1))) {
                occ.add(s.charAt(rk + 1));
                rk++;
            }
            ans = Math.max(ans, rk-i+1);
        }
        return ans;
    }

    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int l1 = nums1.length, l2 = nums2.length;
        int total = l1 + l2;

        if (total % 2 == 1) {
            int mid = total / 2;
            return (double)getKthElement(nums1, nums2, mid+1);
        } else {
            int mid1 = total/2-1, mid2 = total/2;
            return (getKthElement(nums1, nums2, mid1+1) + getKthElement(nums1, nums2, mid2+1)) / 2.0;
        }
    }


    public int getKthElement(int[] nums1, int[] nums2, int k) {
        int length1 = nums1.length, length2 = nums2.length;
        int index1 = 0, index2 = 0;
        int kthElement = 0;

        while (true) {
            // 边界情况
            if (index1 == length1) {
                return nums2[index2 + k - 1];
            }
            if (index2 == length2) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, length1) - 1;
            int newIndex2 = Math.min(index2 + half, length2) - 1;
            int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    // 5. 最长的回文子串
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i=0;i<s.length();i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i+1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end+1);
    }

    public int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    // 10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m+1][n+1];
        f[0][0] = true;
        for (int i=0;i<=m;i++) {
            for (int j=1;j<=n;j++) {
                if (p.charAt(j-1) == '*') {
                    f[i][j] = f[i][j-2];
                    if (matches(s, p, i, j-1)) {
                        f[i][j] = f[i][j] || f[i-1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i-1][j-1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        } else if (p.charAt(j-1) == '.') {
            return true;
        }
        return s.charAt(i-1) == p.charAt(j-1);
    }

    // 11. 盛最多水的容器
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int ans = 0;

        while (left < right) {
            ans = Math.max(ans, Math.min(height[left], height[right]) * (right-left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }

    // 15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();

        for (int first=0;first<n;++first) {
            if (first > 0 && nums[first] == nums[first-1]) {
                continue;
            }
            int third = n - 1;
            int target = -nums[first];

            for (int second=first+1;second<n;++second) {
                if (second > first+1 && nums[second] == nums[second-1]) {
                    continue;
                }

                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    // 17. 电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>();
        if (digits.length() == 0) {
            return ans;
        }
        // String[] phoneMap = {"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        Map<Character, String> phoneMap = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backtrack(ans, phoneMap, digits, 0, new StringBuffer());
        return ans;
    }

    public void backtrack(List<String> combinations,  Map<Character, String> phoneMap, String digits, int index,
                          StringBuffer combination) {
        if (index == digits.length()) {
            combinations.add(combination.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = phoneMap.get(digit);
            int lettersCount = letters.length();
            for (int i = 0; i < lettersCount; i++) {
                combination.append(letters.charAt(i));
                backtrack(combinations, phoneMap, digits, index + 1, combination);
                combination.deleteCharAt(index);
            }
        }
    }

    // 19. 删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head, second = dummy;

        for (int i=0;i<n;i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        ListNode ans = dummy.next;
        return ans;
    }

    // 20. 有效的括号
    public boolean isValid(String s) {
        int n = s.length();
        if (n % 2 != 0) {
            return false;
        }
        Map<Character, Character> pairs = new HashMap<>();
        pairs.put(')', '(');
        pairs.put(']', '[');
        pairs.put('}', '{');

        Deque<Character> stack = new LinkedList<>();

        for (int i=0;i<n;i++) {
            char ch = s.charAt(i);
            if (pairs.containsKey(ch)) {
                if (stack.isEmpty() || stack.peek() != pairs.get(ch)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(ch);
            }
        }
        return stack.isEmpty();
    }

    // 21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode head = dummy;

        while (l1 != null || l2 != null) {
            if (l1 == null) {
                head.next = l2;
                return dummy.next;
            }
            if (l2 == null) {
                head.next = l1;
                return dummy.next;
            }
            if (l1.val <= l2.val) {
                head.next = l1;
                l1 = l1.next;
            } else {
                head.next = l2;
                l2 = l2.next;
            }
            head = head.next;
        }
        return dummy.next;
    }

    // 22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();

        backtrack(ans, n, "", 0, 0);
        return ans;
    }

    public void backtrack(List<String> ans, int n, String temp, int lb, int rb) {
        if (temp.length() == n * 2) {
            ans.add(temp);
        } else {
            if (lb < n) {
                backtrack(ans, n, temp + "(", lb + 1, rb);
            }
            if (rb < lb) {
                backtrack(ans, n, temp + ")", lb, rb + 1);
            }
        }
    }

    // 23. 合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        // 优先队列
        PriorityQueue<ListNode> q = new PriorityQueue<>((x, y) -> x.val-y.val);

        for (ListNode node : lists) {
            if (node != null) {
                q.add(node);
            }
        }
        ListNode head = new ListNode(0);
        ListNode tail = head;
        while (!q.isEmpty()) {
            tail.next = q.poll();
            tail = tail.next;
            if (tail.next != null) {
                q.add(tail.next);
            }
        }
        return head.next;
    }

    // 31. 下一个排列
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }

    // 32. 最长有效括号
    public int longestValidParentheses(String s) {
        int n = s.length(), maxans = 0;
        int[] dp = new int[n];

        dp[0] = 0;

        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i-1) == '(') {
                    dp[i] = (i >= 2 ? dp[i-2] : 0) + 2;
                } else if (i - dp[i-1] > 0 &&  s.charAt(i - dp[i-1] - 1) == '(') {
                    dp[i] = dp[i-1] + ((i - dp[i-1]) >= 2 ? dp[i - dp[i-1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }

    // 33. 搜索旋转排序数组
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            } else {
                if (nums[mid] >= nums[left] && (target > nums[mid] || target < nums[left])) {
                    left = mid + 1;
                } else if (nums[mid] < nums[left] && target > nums[mid] && target < nums[left]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        int[] ans = new int[2];

        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                l = mid;
                r = mid;
                while (l >= 0 && nums[l] == target) {
                    l--;
                }
                while (r <= nums.length-1 && nums[r] == target) {
                    r++;
                }
                ans[0] = ++l;
                ans[1] = ++r;
                return ans;
            } else {
                if (nums[mid] < target) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        ans[0] = -1;
        ans[1] = -1;
        return ans;
    }

    // 39. 组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();

        dfs(candidates, target, ans, combine, 0);
        return ans;
    }

    public void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        dfs(candidates, target, ans, combine, idx+1);
        if (target - candidates[idx] >= 0) {
            combine.add(candidates[idx]);
            dfs(candidates, target-candidates[idx], ans, combine, idx);
            combine.remove(combine.size() - 1);
        }
    }

    // 42. 接雨水
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0) {
            return 0;
        }

        int[] leftMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = Math.max(leftMax[i-1], height[i]);
        }
        int[] rightMax = new int[n];
        rightMax[n-1] = height[n-1];
        for (int i = n-2; i >= 0; i--) {
            rightMax[i] = Math.max(rightMax[i+1], height[i]) - height[i];
        }

        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += Math.min(leftMax[i], rightMax[i]);
        }
        return ans;
    }

    // 42. 接雨水 -- 单调栈
    public int trapStack(int[] height) {
        int ans = 0, n = height.length;
        Deque<Integer> stack = new LinkedList<>();

        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int left = stack.peek();
                int currWidth = i - left - 1;
                int currHeight = Math.min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stack.push(i);
        }
        return ans;
    }

    // 46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        // 使用一个动态数组保存所有可能的全排列
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        boolean[] used = new boolean[len];
        Deque<Integer> path = new ArrayDeque<>(len);

        dfs(nums, len, 0, path, used, res);
        return res;
    }

    private void dfs(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {
        if (depth == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < len; i++) {
            if (!used[i]) {
                path.addLast(nums[i]);
                used[i] = true;

                System.out.println("  递归之前 => " + path);
                dfs(nums, len, depth + 1, path, used, res);

                used[i] = false;
                path.removeLast();
                System.out.println("递归之后 => " + path);
            }
        }
    }

    // 48. 旋转图像
    public void rotate(int[][] matrix) {
        int row = matrix.length;

        for (int i = 0; i < row/2; i++) {
            for (int j = 0; j < (row+1)/2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[row - j - 1][i];
                matrix[row - j - 1][i] = matrix[row - i - 1][row - j - 1];
                matrix[row - i - 1][row - j - 1] = matrix[j][row - i - 1];
                matrix[j][row - i - 1] = temp;
            }
        }
    }

    // 49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }

    // 53. 最大子序和
    public int maxSubArray(int[] nums) {
        int ans = nums[0], sum = 0;
        for(int num: nums) {
            if(sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            ans = Math.max(ans, sum);
        }
        return ans;
    }

    // 55. 跳跃游戏
    public boolean canJump(int[] nums) {
        int n = nums.length, rmax = 0;
        for (int i = 0; i < n; i++) {
            if (i <= rmax) {
                rmax = Math.max(rmax, i + nums[i]);
                if (rmax >= n - 1) {
                    return true;
                }
            } else {
                return false;
            }
        }
        return false;
    }

    // 56. 合并区间
    // 重写了 Arrays.sort(intervals, new Comparator<int[]>() {...});
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        List<int[]> merged = new ArrayList<>();
        for (int i = 0; i < intervals.length; i++) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    // 62. 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        dp[0][0] = 1;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }

    // 64. 最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];

        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
            }
        }

        return dp[m-1][n-1];
    }

    // 70. 爬楼梯
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i < n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n-1];
    }

    // 72. 编辑距离
    public int minDistance(String word1, String word2) {
        // 定理一：如果其中一个字符串是空串，那么编辑距离是另一个字符串的长度。比如空串“”和“ro”的编辑距离是2（做两次“插入”操作）。再比如"hor"和空串“”的编辑距离是3（做三次“删除”操作）。
        //
        // 定理二：当i>0,j>0时（即两个串都不空时）dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+int(word1[i]!=word2[j]))。
        //
        // 啥意思呢？举个例子，word1 = "abcde", word2 = "fgh",我们现在算这俩字符串的编辑距离，就是找从word1，最少多少步，能变成word2？那就有三种方式：
        //
        // 知道"abcd"变成"fgh"多少步（假设X步），那么从"abcde"到"fgh"就是"abcde"->"abcd"->"fgh"。（一次删除，加X步，总共X+1步）
        // 知道"abcde"变成“fg”多少步（假设Y步），那么从"abcde"到"fgh"就是"abcde"->"fg"->"fgh"。（先Y步，再一次添加，加X步，总共Y+1步）
        // 知道"abcd"变成“fg”多少步（假设Z步），那么从"abcde"到"fgh"就是"abcde"->"fge"->"fgh"。（先不管最后一个字符，把前面的先变好，用了Z步，然后把最后一个字符给替换了。这里如果最后一个字符碰巧就一样，那就不用替换，省了一步）
        int n = word1.length(), m = word2.length();
        if (n * m == 0) {
            return n + m;
        }
        int [][] dp = new int[n+1][m+1];

        // 边界状态初始化
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < m + 1; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i < n + 1; i++) {
            for ( int j = 1; j < m + 1; j++) {
                int left = dp[i - 1][j] + 1;
                int down = dp[i][j - 1] + 1;
                int left_down = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    left_down += 1;
                }
                dp[i][j] = Math.min(left, Math.min(down, left_down));
            }
        }
        return dp[n][m];
    }

    // 75. 颜色分类
    public void sortColors(int[] nums) {
        int n = nums.length;
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            while (i <= p2 && nums[i] == 2) {
                int temp = nums[i];
                nums[i] = nums[p2];
                nums[p2] = temp;
                --p2;
            }
            if (nums[i] == 0) {
                int temp = nums[i];
                nums[i] = nums[p0];
                nums[p0] = temp;
                ++p0;
            }
        }
    }

    // 76. 最小覆盖子串 -- 写的也太好了吧！
    public String minWindow(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0){
            return "";
        }
        int[] need = new int[128];
        //记录需要的字符的个数
        for (int i = 0; i < t.length(); i++) {
            need[t.charAt(i)]++;
        }
        //l是当前左边界，r是当前右边界，size记录窗口大小，count是需求的字符个数，start是最小覆盖串开始的index
        int l = 0, r = 0, size = Integer.MAX_VALUE, count = t.length(), start = 0;
        //遍历所有字符
        while (r < s.length()) {
            char c = s.charAt(r);
            if (need[c] > 0) {//需要字符c
                count--;
            }
            need[c]--;//把右边的字符加入窗口
            if (count == 0) {//窗口中已经包含所有字符
                while (l < r && need[s.charAt(l)] < 0) {
                    need[s.charAt(l)]++;//释放右边移动出窗口的字符
                    l++;//指针右移
                }
                if (r - l + 1 < size) {//不能右移时候挑战最小窗口大小，更新最小窗口开始的start
                    size = r - l + 1;
                    start = l;//记录下最小值时候的开始位置，最后返回覆盖串时候会用到
                }
                //l向右移动后窗口肯定不能满足了 重新开始循环
                need[s.charAt(l)]++;
                l++;
                count++;
            }
            r++;
        }
        return size == Integer.MAX_VALUE ? "" : s.substring(start, start + size);
    }

    // 78. 子集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> stack = new ArrayList<>();

        dfs(nums, stack, res, 0);
        return res;
    }

    public void dfs(int[] nums, List<Integer> stack, List<List<Integer>> res, int index) {
        if (index == nums.length) {
            res.add(new ArrayList<>(stack));
            return;
        }

        stack.add(nums[index]);
        dfs(nums, stack, res, index+1);
        stack.remove(stack.size() - 1);
        dfs(nums, stack, res, index+1);
    }

    // 79. 单词搜索
    public boolean exist(char[][] board, String word) {
        int h = board.length, w = board[0].length;
        boolean[][] visited = new boolean[h][w];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                boolean flag = check(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean check(char[][] board, boolean[][] visited, int i, int j, String s, int k) {
        if (board[i][j] != s.charAt(k)) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        int[][] directions = {{0,1}, {0,-1}, {1,0}, {-1,0}};
        boolean result = false;
        for (int[] dir : directions) {
            int newi = i + dir[0], newj = j + dir[1];
            if (newi >= 0 && newi < board.length && newj >= 0 && newj < board[0].length) {
                if (!visited[newi][newj]) {
                    boolean flag = check(board, visited, newi, newj, s, k + 1);
                    if (flag) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }

    // 84. 柱状图中最大的矩形
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Arrays.fill(right, n);

        Stack<Integer> mono_stack = new Stack<>();
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                right[mono_stack.peek()] = i;
                mono_stack.pop();
            }
            left[i] = (mono_stack.isEmpty() ? -1 : mono_stack.peek());
            mono_stack.push(i);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = Math.max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }

    // 85. 最大矩形
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0) {
            return 0;
        }
        int[] heights = new int[matrix[0].length];
        int maxArea = 0;
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                if (matrix[row][col] == '1') {
                    heights[col] += 1;
                } else {
                    heights[col] = 0;
                }
            }
            // 调用 84 的解法
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }
        return maxArea;
    }

    // 94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<Integer> stack = new Stack<>();

        inorder(root, stack);
        return stack;
    }

    public void inorder(TreeNode root, Stack<Integer> stack) {
        if (root != null) {
            inorder(root.left, stack);
            stack.push(root.val);
            inorder(root.right, stack);
        }
    }

    // 96. 不同的二叉搜索树
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    // 98. 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        return inorder(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    public boolean inorder(TreeNode root, long lower, long upper) {
        if (root == null) {
            return true;
        }
        if (root.val <= lower || root.val >= upper) {
            return false;
        }
        return inorder(root.left, lower, root.val) && inorder(root.right, root.val, upper);
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return check(root.left, root.right);
    }

    public boolean check(TreeNode lnode, TreeNode rnode) {
        if (lnode == null && rnode == null) {
            return true;
        } else if (lnode == null || rnode == null) {
            return false;
        } else {
            return lnode.val == rnode.val && check(lnode.left, rnode.right) && check(lnode.right, rnode.left);
        }
    }

    // 102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 1; i <= size; i++) {
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            res.add(level);
        }
        return res;
    }

    // 104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        return dfs(root, 1);
    }

    public int dfs(TreeNode root, int deep) {
        if (root == null) {
            return deep;
        }
        return Math.max(dfs(root.left, deep+1), dfs(root.right, deep+1));
    }

    // 105. 从前序与中序遍历序列构造二叉树

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, 0, preorder.length, inorder, 0, inorder.length);
    }

    private TreeNode buildTreeHelper(int[] preorder, int p_start, int p_end, int[] inorder, int i_start, int i_end) {
        // preorder 为空，直接返回 null
        if (p_start == p_end) {
            return null;
        }
        int root_val = preorder[p_start];
        TreeNode root = new TreeNode(root_val);
        //在中序遍历中找到根节点的位置
        int i_root_index = 0;
        for (int i = i_start; i < i_end; i++) {
            if (root_val == inorder[i]) {
                i_root_index = i;
                break;
            }
        }
        int leftNum = i_root_index - i_start;
        //递归的构造左子树
        root.left = buildTreeHelper(preorder, p_start + 1, p_start + leftNum + 1, inorder, i_start, i_root_index);
        //递归的构造右子树
        root.right = buildTreeHelper(preorder, p_start + leftNum + 1, p_end, inorder, i_root_index + 1, i_end);
        return root;
    }

    // 114. 二叉树展开为链表
    public void flatten(TreeNode root) {
        while (root != null) {
            if (root.left == null) {
                root = root.right;
            } else {
                TreeNode pre = root.left;
                while (pre.right != null) {
                    pre = pre.right;
                }
                pre.right = root.right;
                root.right = root.left;
                root.left = null;
                root = root.right;
            }
        }
    }

    // 121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int min = prices[0];
        int ans = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < min) {
                min = prices[i];
            } else {
                ans = Math.max(ans, prices[i] - min);
            }
        }
        return ans;
    }

    // 124. 二叉树中的最大路径和
    int maxRes = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        postOrder(root);
        return maxRes;
    }

    // 拿到一边，包括该root 的最大sum
    private int postOrder(TreeNode root){
        if(root == null)
            return 0;

        int left = Math.max(postOrder(root.left), 0);
        int right = Math.max(postOrder(root.right), 0);

        maxRes = Math.max(maxRes, root.val + left + right);

        return root.val + Math.max(left, right);
    }

    // 128. 最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        int maxLength = 0;

        for (int num : numSet) {
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentLength = 1;

                while (numSet.contains(currentNum + 1)) {
                    currentNum++;
                    currentLength++;
                }
                maxLength = Math.max(currentLength, maxLength);
            }
        }
        return maxLength;
    }

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            ans ^= nums[i];
        }
        return ans;
    }

    // 139. 单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    // 141. 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head, fast = head.next;
        while (fast != slow) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    // 142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
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

    // 146. LRU 缓存机制

    // 148. 排序链表
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left != null && right != null) {
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        h.next = left != null ? left : right;
        return res.next;
    }

    // 152. 乘积最大子数组
    public int maxProduct(int[] nums) {
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        int length = nums.length;
        for (int i = 1; i < nums.length; i++) {
            int mx = maxF, mn = minF;
            maxF = Math.max(mx * nums[i], Math.max(nums[i], mn * nums[i]));
            minF = Math.min(mn * nums[i], Math.min(nums[i], mx * nums[i]));
            ans = Math.max(maxF, ans);
        }
        return ans;
    }

    // 155. 最小栈

    // 160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode ptr1 = headA;
        ListNode ptr2 = headB;

        while (ptr1 != ptr2) {
            if (ptr1 == null) {
                ptr1 = headB;
            } else {
                ptr1 = ptr1.next;
            }
            if (ptr2 == null) {
                ptr2 = headA;
            } else {
                ptr2 = ptr2.next;
            }
        }
        return ptr1;
    }

    // 169. 多数元素
    public int majorityElement(int[] nums) {
        int ans = 0, count = 0;

        for (int num : nums) {
            if (num == ans) {
                count++;
            } else {
                if (count != 0) {
                    count--;
                } else {
                    ans = num;
                    count++;
                }
            }
        }
        return ans;
    }

    // 198. 打家劫舍
    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        } else if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
        }
        return dp[nums.length-1];
    }

    // 200. 岛屿数量
    public int numIslands(char[][] grid) {
        int ans = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j, grid.length, grid[0].length);
                    ans++;
                }
            }
        }
        return ans;
    }

    public void dfs(char[][]grid, int i, int j, int row, int col) {
        if (i >= 0 && i < row && j >= 0 && j < col && grid[i][j] == '1') {
            grid[i][j] = '0';
            dfs(grid, i-1, j, row, col);
            dfs(grid, i+1, j, row, col);
            dfs(grid, i, j-1, row, col);
            dfs(grid, i, j+1, row, col);
        }
    }

    // 206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode prev = null, node = head;
        while (node != null) {
            ListNode tmp = node.next;
            node.next = prev;
            prev = node;
            node = tmp;
        }
        return head;
    }

    // 207. 课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjacency.add(new ArrayList<>());
        }
        int[] flags = new int[numCourses];
        for (int[] cp : prerequisites) {
            adjacency.get(cp[1]).add(cp[0]);
        }
        for (int i = 0; i < numCourses; i++) {
            if (courseDfs(adjacency, flags, i)) {
                return false;
            }
        }
        return true;
    }

    private boolean courseDfs(List<List<Integer>> adjacency, int[] flags, int i) {
        if(flags[i] == 1) return true;
        if(flags[i] == -1) return false;
        flags[i] = 1;
        for(Integer j : adjacency.get(i))
            if(courseDfs(adjacency, flags, j)) return true;
        flags[i] = -1;
        return false;
    }

    // 208. 实现 Trie (前缀树)

    // 215. 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        int len = nums.length;
        int target = len - k;
        int left = 0, right = len - 1;
        while (true) {
            int index = partition(nums, left, right);
            if (index < target) {
                left = index + 1;
            } else if (index > target) {
                right = index - 1;
            } else {
                return nums[index];
            }
        }
    }

    private int partition(int[] nums, int left, int right) {
        Random random = new Random(System.currentTimeMillis());
        if (right > left) {
            int randomIndex = left + 1 + random.nextInt(right-left);
            swap(nums, left, randomIndex);
        }

        int pivot = nums[left];
        int j = left;
        for (int i = left + 1; i <= right; i++) {
            if (nums[i] < pivot) {
                j++;
                swap(nums, j, i);
            }
        }
        swap(nums, left, j);
        return j;
    }

    // 221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        int[][] dp = new int[row][col];
        int ans = 0;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == '1') {
                    if (i * j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(dp[i-1][j], Math.min(dp[i][j-1], dp[i-1][j-1])) + 1;
                    }
                    ans = Math.max(ans, dp[i][j]);
                }
            }
        }
        return (int)Math.pow(ans, 2);
    }

    // 226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root != null) {
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;

            invertTree(root.left);
            invertTree(root.right);
        }
        return root;
    }

    // 234. 回文链表
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode slow = head, fast = head;
        ListNode pre = head, prepre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
            pre.next = prepre;
            prepre = pre;
        }
        if (fast != null) {
            slow = slow.next;
        }
        while (pre != null && slow != null) {
            if (pre.val != slow.val) {
                return false;
            }
            pre = pre.next;
            slow = slow.next;
        }
        return true;
    }

    // 236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
             return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        if (left == null) {
            return right;
        }
        return left;
    }

    // 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] ans = new int[len];
        ans[0] = 1;

        for (int i = 1; i < len; i++) {
            ans[i] = nums[i-1] * ans[i-1];
        }

        int r = 1;
        for (int i = len-2; i >= 0 ; i--) {
            r = nums[i+1] * r;
            ans[i] *= r;
        }

        return ans;
    }

    // 239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length < 2) return nums;
        LinkedList<Integer> queue = new LinkedList<>();
        int[] result = new int[nums.length-k+1];

        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && nums[queue.peekLast()] <= nums[i]) {
                queue.pollLast();
            }
            queue.addLast(i);
            if (queue.peek() <= i - k) {
                queue.poll();
            }
            if (i + 1 >= k) {
                result[i+1-k] = nums[queue.peek()];
            }
        }
        return result;
    }

    // 240. 搜索二维矩阵 II
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length, col = matrix[0].length;
        int i = row - 1, j = 0;
        while (i >=0 && j < col) {
            if (target > matrix[i][j]) {
                j++;
            } else if (target < matrix[i][j]) {
                i--;
            } else {
                return true;
            }
        }
        return false;
    }

    // 253. 会议室 II

    // 279. 完全平方数
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        for (int i = 0;i < n+1; i++) {
            dp[i] = i;
        }

        for (int i = 1; i <= n; i++) {
            int index = 1;
            while ((i - index * index) >= 0) {
                dp[i] = Math.min(dp[i], dp[i-index * index] + 1);
                index++;
            }
        }
        return dp[n];
    }

    // 283. 移动零
    public void moveZeroes(int[] nums) {
        int n = nums.length, left = 0, right = 0;
        while (right < n) {
            if (nums[right] != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;
        }
    }

    // 287. 寻找重复数
    public int findDuplicate(int[] nums) {
        int len = nums.length, left = 1, right = len - 1;
        while (left < right) {
            // 在 Java 里可以这么用，当 left + right 溢出的时候，无符号右移保证结果依然正确
            int mid = (left + right) >>> 1;
            int cnt = 0;
            for (int num : nums) {
                if (num <= mid) {
                    cnt++;
                }
            }

            if (cnt > mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // 297. 二叉树的序列化与反序列化

    // 300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int ans = 1;
        tails[0] = nums[0];

        for (int num : nums) {
            if (num > tails[ans-1]) {
                tails[ans] = num;
                ans++;
            } else {
                int i = 0, j = ans;
                while (i < j) {
                    int m = (i + j) >>> 1;

                    if (tails[m] < num) {
                        i = m + 1;
                    } else {
                        j = m;
                    }
                }
                tails[i] = num;
            }
        }
        return ans;
    }

    // 301. 删除无效的括号
    public List<String> removeInvalidParentheses(String s) {
        int rml = 0, rmr = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                rml++;
            } else if (c == ')') {
                if (rml > 0) {
                    rml--;
                } else {
                    rmr++;
                }
            }
        }
        HashSet<String> result = new HashSet<>();
        backtrack(result, new StringBuilder(), s, 0, 0, 0, rml, rmr);
        return new ArrayList<>(result);
    }

    private void backtrack(HashSet<String> result, StringBuilder temp, String s, int index, int lcount, int rcount,
                           int rml, int rmr) {
        if (index == s.length()) {
            if (rml == 0 && rmr == 0) {
                result.add(temp.toString());
            }
            return;
        }

        char character = s.toCharArray()[index];
        if (character == '(' && rml > 0) {
            backtrack(result, temp, s, index + 1, lcount, rcount, rml - 1, rmr);
        } else if (character == ')' && rmr > 0) {
            backtrack(result, temp, s, index + 1, lcount, rcount, rml, rmr - 1);
        }

        temp.append(character);
        if (character != '(' && character != ')') {
            backtrack(result, temp, s, index + 1, lcount, rcount, rml, rmr);
        } else if (character == '(') {
            backtrack(result, temp, s, index + 1, lcount + 1, rcount, rml, rmr);
        } else if (rcount < lcount) {  // 如果在中间的任何时刻右括号都不能大于左括号
            backtrack(result, temp, s, index + 1, lcount, rcount + 1, rml, rmr);
        }
        temp.deleteCharAt(temp.length() - 1);
    }

    // 309. 最佳买卖股票时机含冷冻期
    public int maxProfitForce(int[] prices) {
        // 三种状态：持有股票的最大收益，卖出且为冷冻期的最大收益，卖出且不在冷冻期的最大收益
        int[][] dp = new int[prices.length][3];
        dp[0][0] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][2] - prices[i]);
            dp[i][1] = dp[i-1][0] + prices[i];
            dp[i][2] = Math.max(dp[i-1][1], dp[i-1][2]);
        }
        return Math.max(dp[prices.length-1][1], dp[prices.length-1][2]);
    }

    // 312. 戳气球
    public int maxCoins(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        int[] arr = new int[n+2];
        System.arraycopy(nums, 0, arr, 1, n);
        arr[0] = 1;
        arr[n+1] = 1;
        int[][] dp = new int[n+2][n+2];

        for (int i = n-1; i >= 0; i--) {
            for (int j = i+2; j <= n+1; j++) {
                // k在(i,j)范围内遍历气球，计算每选择一个气球的积分
                // 一轮遍历完后，就能确定(i,j)的最大积分
                for(int k = i + 1; k < j; ++k) {
                    int total = arr[i] * arr[k] * arr[j];
                    total += dp[i][k] + dp[k][j];
                    dp[i][j] = Math.max(dp[i][j], total);
                }
            }
        }
        return dp[0][n+1];
    }

    // 322. 零钱兑换
    public int coinChange(int[] coins, int amount) {
        int len = coins.length;
        int[] dp = new int[amount + 1];
        for (int i = 1; i < amount + 1; i++) {
            dp[i] = -1;
        }
        Arrays.sort(coins);
        for (int i = 0; i <= amount; i++) {
            for (int coin : coins) {
                if (i - coin < 0) {
                    break;
                } else {
                    if (dp[i - coin] == -1) {
                        continue;
                    }
                    dp[i] = dp[i] == -1 ? dp[i - coin] + 1 : Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount];
    }

    // 337. 打家劫舍 III
    public int rob(TreeNode root) {
        int[] result = robInternal(root);
        return Math.max(result[0], result[1]);
    }

    private int[] robInternal(TreeNode root) {
        if (root == null) return new int[2];
        int[] result = new int[2];

        int[] left = robInternal(root.left);
        int[] right = robInternal(root.right);

        result[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        result[1] = root.val + left[0] + right[0];

        return result;
    }

    // 338. 比特位计数
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        int highBit = 0;
        for (int i = 1; i <= num; i++) {
            if ((i & (i - 1)) == 0) {
                highBit = i;
            }
            bits[i] = bits[i - highBit] + 1;
        }
        return bits;
    }

    // 347. 前 K 个高频元素 Top K
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        PriorityQueue<int []> queue = new PriorityQueue<int []>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            if (queue.size() == k) {
                assert queue.peek() != null;
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = Objects.requireNonNull(queue.poll())[0];
        }
        return ret;
    }

    // 394. 字符串解码
    public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();

        for (Character c : s.toCharArray()) {
            if (c == '[') {
                stack_multi.addLast(multi);
                stack_res.addLast(res.toString());
                multi = 0;
                res = new StringBuilder();
            } else if (c == ']') {
                StringBuilder tmp = new StringBuilder();
                int cur_multi = stack_multi.removeLast();
                for (int i = 0; i < cur_multi; i++) tmp.append(res);
                res = new StringBuilder(stack_res.removeLast() + tmp);
            } else if (c >= '0' && c <= '9') {
                multi = multi * 10 + Integer.parseInt(c + ""); // '+ ""' for convert char to String
            } else {
                res.append(c);
            }
        }
        return res.toString();
    }

    // 399. 除法求值
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int count=0;
        //统计出现的所有字符，并赋予对应的index
        Map<String,Integer> map=new HashMap<>();
        for (List<String> list:equations){
            for (String s:list){
                if(!map.containsKey(s)){
                    map.put(s,count++);
                }
            }
        }

        //构建一个矩阵来代替图结构
        double[][] graph=new double[count+1][count+1];

        //初始化
        for (String s:map.keySet()){
            int x=map.get(s);
            graph[x][x]=1.0;
        }
        int index=0;
        for (List<String> list:equations){
            String a=list.get(0);
            String b=list.get(1);
            int aa=map.get(a);
            int bb=map.get(b);
            double value=values[index++];
            graph[aa][bb]=value;
            graph[bb][aa]=1/value;
        }

        //通过Floyd算法进行运算
        int n=count+1;
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                for (int k=0;k<n;k++){
                    if(j==k||graph[j][k]!=0) continue;
                    if(graph[j][i]!=0&&graph[i][k]!=0){
                        graph[j][k]=graph[j][i]*graph[i][k];
                    }
                }
            }
        }

        //直接通过查询矩阵得到答案
        double[] res=new double[queries.size()];
        for (int i=0;i<res.length;i++){
            List<String> q=queries.get(i);
            String a=q.get(0);
            String b=q.get(1);
            if(map.containsKey(a)&&map.containsKey(b)){
                double ans=graph[map.get(a)][map.get(b)];
                res[i]=(ans==0?-1.0:ans);
            }else {
                res[i]=-1.0;
            }
        }
        return res;
    }

    // 406. 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) {
                    return o1[0] - o2[0];
                } else {
                    return o2[1] - o1[1];
                }
            }
        });
        int n = people.length;
        int[][] ans = new int[n][];
        for (int[] person : people) {
            int spaces = person[1] + 1;
            for (int i = 0; i < n; ++i) {
                if (ans[i] == null) {
                    --spaces;
                    if (spaces == 0) {
                        ans[i] = person;
                        break;
                    }
                }
            }
        }
        return ans;
    }

    // 416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum % 2 == 1 || nums.length == 1) {
            return false;
        }
        int target = sum / 2;
        for (int num :nums) {
            if (num > target) {
                return false;
            }
        }

        boolean[][] dp = new boolean[nums.length][target + 1];

        if (nums[0] <= target) {
            dp[0][nums[0]] = true;
        }
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (nums[i] == j) {
                    dp[i][j] = true;
                    continue;
                }
                if (nums[i] < j) {
                    dp[i][j] = dp[i-1][j] || dp[i-1][j - nums[i]];
                }
            }
        }
        return dp[nums.length - 1][target];
    }

    // 437. 路径总和 III
    public int pathSum(TreeNode root, int targetSum) {
        // key是前缀和, value是大小为key的前缀和出现的次数
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        // 前缀和为0的一条路径
        prefixSumCount.put(0, 1);
        // 前缀和的递归回溯思路
        return recursionPathSum(root, prefixSumCount, targetSum, 0);
    }

    private int recursionPathSum(TreeNode node, Map<Integer, Integer> prefixSumCount, int target, int currSum) {
        if (node == null) {
            return 0;
        }
        int res = 0;
        currSum += node.val;

        res += prefixSumCount.getOrDefault(currSum- target, 0);
        prefixSumCount.put(currSum, prefixSumCount.getOrDefault(currSum, 0) + 1);

        res += recursionPathSum(node.left, prefixSumCount, target, currSum);
        res += recursionPathSum(node.right, prefixSumCount, target, currSum);

        // 4.回到本层，恢复状态，去除当前节点的前缀和数量
        prefixSumCount.put(currSum, prefixSumCount.get(currSum) - 1);
        return res;
    }

    // 438. 找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        int slen = s.length(), plen = p.length();
        Map<Character, Integer> charmap = new HashMap<>();
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < plen; i++) {
            char ch = p.charAt(i);
            charmap.put(ch, charmap.getOrDefault(ch, 0) - 1);
        }

        for (int i = 0; i < slen; i++) {
            char ch = s.charAt(i);

            charmap.put(ch, charmap.getOrDefault(ch, 0) + 1);
            if (i >= plen) {
                char chl = s.charAt(i - plen);
                charmap.put(chl, charmap.getOrDefault(chl, 0) - 1);
            }
            if (check(charmap)) {
                res.add(i- plen + 1);
            }
        }
        return res;
    }

    private boolean check(Map<Character, Integer> charmap) {
        boolean flag = true;
        for (Map.Entry<Character, Integer> i : charmap.entrySet()) {
            if (i.getValue() != 0) {
                flag = false;
                break;
            }
        }
        return flag;
    }

    // 448. 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1) % n;
            nums[x] += n;
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                res.add(i);
            }
        }
        return res;
    }

    // 461. 汉明距离
    public int hammingDistance(int x, int y) {
        int ans = 0, num = x^y;
        while (num != 0) {
            if ((num & 1) == 1) {
                ans++;
            }
            num >>= 1;
        }
        return ans;
    }

    // 494. 目标和
    public int findTargetSumWays(int[] nums, int target) {
        int[][] dp = new int[nums.length][2001];
        dp[0][nums[0] + 1000] = 1;
        dp[0][-nums[0] + 1000] += 1;

        for (int i = 1; i < nums.length; i++) {
            for(int sum = -1000; sum <= 1000; sum++) {
                if (dp[i - 1][sum + 1000] > 0) {
                    dp[i][sum + nums[i] + 1000] += dp[i - 1][sum + 1000];
                    dp[i][sum - nums[i] + 1000] += dp[i - 1][sum + 1000];
                }
            }
        }
        return target > 1000 ? 0 : dp[nums.length - 1][target + 1000];
    }

    // 538. 把二叉搜索树转换为累加树
    public TreeNode convertBST(TreeNode root) {
        int sum = dfsBST(root, 0);
        return root;
    }

    private int dfsBST(TreeNode root, int sum) {
        if (root != null) {
            sum = dfsBST(root.right, sum);
            sum += root.val;
            root.val = sum;
            sum = dfsBST(root.left, sum);
        }
        return sum;
    }

    // 543. 二叉树的直径
    int ans;
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }

    public int depth(TreeNode node) {
        if (node == null) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L+R+1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }

    // 560. 和为K的子数组
    public int subarraySum(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        mp.put(0, 1);
        for (int i = 0; i < nums.length; i++){
            pre += nums[i];
            if (mp.containsKey(pre-k)) {
                count += mp.get(pre-k);
            }
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;
    }

    // 581. 最短无序连续子数组
    public int findUnsortedSubarray(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int l = nums.length, r = 0;
        for (int i = 0; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                l = Math.min(l, stack.pop());
            }
            stack.push(i);
        }
        stack.clear();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i])
                r = Math.max(r, stack.pop());
            stack.push(i);
        }
        return r - l > 0 ? r - l + 1 : 0;
    }

    // 617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 != null) {
            return root2;
        } else if (root1 != null && root2 == null) {
            return root1;
        } else if (root1 == null && root2 == null) {
            return null;
        } else {
            root1.val += root2.val;
            root1.left = mergeTrees(root1.left, root2.left);
            root1.right = mergeTrees(root1.right, root2.right);
            return root1;
        }
    }

    // 621. 任务调度器
    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> freq = new HashMap<Character, Integer>();
        // 最多的执行次数
        int maxExec = 0;
        for (char ch : tasks) {
            int exec = freq.getOrDefault(ch, 0) + 1;
            freq.put(ch, exec);
            maxExec = Math.max(maxExec, exec);
        }

        // 具有最多执行次数的任务数量
        int maxCount = 0;
        Set<Map.Entry<Character, Integer>> entrySet = freq.entrySet();
        for (Map.Entry<Character, Integer> entry : entrySet) {
            int value = entry.getValue();
            if (value == maxExec) {
                ++maxCount;
            }
        }

        return Math.max((maxExec - 1) * (n + 1) + maxCount, tasks.length);
    }

    // 647. 回文子串
    public int countSubstrings(String s) {
        int n = s.length(), ans = 0;
        for (int i = 0; i < 2 * n - 1; ++i) {
            int l = i / 2, r = i / 2 + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                --l;
                ++r;
                ++ans;
            }
        }
        return ans;
    }

    // 739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        int[] ans = new int[temperatures.length];
        List<Integer> stack = new ArrayList<>();

        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[stack.get(stack.size() - 1)] < temperatures[i]) {
                ans[stack.get(stack.size() - 1)] = i - stack.get(stack.size() - 1);
                stack.remove(stack.size() - 1);
            }
            stack.add(i);
        }

        return ans;
    }

    // =============== 每日一题 ===============

    // 1035. 不相交的线
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int n = nums1.length, m = nums2.length;
        int[][] dp = new int[n][m];

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                if (nums1[i-1] == nums2[j-1]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i-1][j-1] + 1);
                }
            }
        }
        return dp[n-1][m-1];
    }

    // 810. 黑板异或游戏  hard
    public boolean xorGame(int[] nums) {
        if (nums.length % 2 == 0) {
            return true;
        }
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        return xor == 0;
    }

    // 692. 前K个高频单词  hard
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> cnt = new HashMap<>();
        for (String word : words) {
            cnt.put(word, cnt.getOrDefault(word, 0) + 1);
        }
        List<String> rec = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : cnt.entrySet()) {
            rec.add(entry.getKey());
        }
        // 手写排序，相等则比较字典序，不相等则比较哈希表中的出现次数
        Collections.sort(rec, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return cnt.get(o1).equals(cnt.get(o2)) ? o1.compareTo(o2) : cnt.get(o2) - cnt.get(o1);
            }
        });
        return rec.subList(0, k);
    }

    // 1707. 与数组中元素的最大异或值  hard
    public int[] maximizeXor(int[] nums, int[][] queries) {
        Trie trie = new Trie();
        for (int val : nums) {
            trie.insert(val);
        }
        int numQ = queries.length;
        int[] ans = new int[numQ];
        for (int i = 0; i < numQ; ++i) {
            ans[i] = trie.getMaxXorWithLimit(queries[i][0], queries[i][1]);
        }
        return ans;
    }

    static class Trie {
        static final int L = 30;
        Trie[] children = new Trie[2];
        int min = Integer.MAX_VALUE;

        public void insert(int val) {
            Trie node = this;
            node.min = Math.min(node.min, val);
            for (int i = L - 1; i >= 0; --i) {
                int bit = (val >> i) & 1;
                if (node.children[bit] == null) {
                    node.children[bit] = new Trie();
                }
                node = node.children[bit];
                node.min = Math.min(node.min, val);
            }
        }

        public int getMaxXorWithLimit(int val, int limit) {
            Trie node = this;
            if (node.min > limit) {
                return -1;
            }
            int ans = 0;
            for (int i = L - 1; i >= 0; --i) {
                int bit = (val >> i) & 1;
                if (node.children[bit ^ 1] != null && node.children[bit ^ 1].min <= limit) {
                    ans |= 1 << i;
                    bit ^= 1;
                }
                node = node.children[bit];
            }
            return ans;
        }
    }

    // 664. 奇怪的打印机  hard
    public int strangePrinter(String s) {
        int n = s.length();
        int[][] f = new int[n][n];
        for (int i = n-1; i >= 0; i--) {
            f[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    f[i][j] = f[i][j - 1];
                } else {
                    int minn = Integer.MAX_VALUE;
                    for (int k = i; k < j; k++) {
                        minn = Math.min(minn, f[i][k] + f[k + 1][j]);
                    }
                    f[i][j] = minn;
                }
            }
        }
        return f[0][n-1];
    }

    // 1787. 使所有区间的异或结果为零  hard
    public int minChanges(int[] nums, int k) {
        int n = nums.length, m = 1024;
        int[] prev = new int[m];
        int prevMin = Integer.MAX_VALUE;
        for (int i = 0; i < k; ++i) {// O(K)
            int total = 0;
            int[] counter = new int[m];
            for (int l = i; l < n; l += k, ++total) ++counter[nums[l]];
            if (i == 0) {
                for (int j = 0; j < m; ++j) {
                    prev[j] = total - counter[j];
                    prevMin = Math.min(prevMin, prev[j]);
                }
            } else {
                int[] curr = new int[m];
                int currMin = Integer.MAX_VALUE;
                for (int j = 0; j < m; ++j) {// O(1024)
                    curr[j] = prevMin + total;
                    for (int l = i; l < n; l += k) {// O(N/K)
                        curr[j] = Math.min(curr[j], prev[j ^ nums[l]] + total - counter[nums[l]]);
                    }
                    currMin = Math.min(currMin, curr[j]);
                }
                prev = curr;
                prevMin = currMin;
            }
        }
        return prev[0];
    }

    // 1190. 反转每对括号间的子串
    public String reverseParentheses(String s) {
        Deque<String> stack = new LinkedList<>();
        StringBuffer sb = new StringBuffer();

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                stack.push(sb.toString());
                sb.setLength(0);
            } else if (ch == ')') {
                sb.reverse();
                sb.insert(0, stack.pop());
            } else {
                sb.append(ch);
            }
        }
        return sb.toString();
    }

    // 477. 汉明距离总和
    public int totalHammingDistance(int[] nums) {
        int ans = 0;
        for (int x = 31; x >= 0; x--) {
            int s0 = 0, s1 = 0;
            for (int u : nums) {
                if (((u >> x) & 1) == 1) {
                    s1++;
                } else {
                    s0++;
                }
            }
            ans += s0 * s1;
        }
        return ans;
    }

    // 1074. 元素和为目标值的子矩阵数量 hard 前缀和 + HashMap
    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int n = matrix.length, m = matrix[0].length;
        int[][] sum = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + matrix[i - 1][j - 1];
            }
        }
        int ans = 0;
        for (int top = 1; top <= n; top++) {
            for (int bot = top; bot <= m; bot++) {
                int cur = 0;
                Map<Integer, Integer> map = new HashMap<>();
                for (int r = 1; r <= m; r++) {
                    cur = sum[bot][r] - sum[top - 1][r];
                    if (cur == target) ans++;
                    if (map.containsKey(cur - target)) ans += map.get(cur - target);
                    map.put(cur, map.getOrDefault(cur, 0) + 1);
                }
            }
        }
        return ans;
    }

    // 231. 2 的幂 easy 位运算
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // 342. 4的幂 easy 位运算
    public boolean isPowerOfFour(int n) {
        return (n & (n - 1)) == 0 && n % 3 == 1;
    }

    // 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？ medium
    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        int n = queries.length, m = candiesCount.length;
        boolean[] ans = new boolean[n];
        long[] sum = new long[m + 1];
        for (int i = 1; i <= m; i++) sum[i] = sum[i - 1] + candiesCount[i - 1];
        for (int i = 0; i < n; i++) {
            int t = queries[i][0], d = queries[i][1] + 1, c = queries[i][2];
            long a = sum[t] / c + 1, b = sum[t + 1];
            ans[i] = a <= d && d <= b;
        }
        return ans;
    }

    // 523. 连续的子数组和 medium
    public boolean checkSubarraySum(int[] nums, int k) {
        int m = nums.length;
        int[] sum = new int[m + 1];

        for (int i = 1; i < m + 1; i++) sum[i] = sum[i-1] + nums[i-1];
        Set<Integer> set = new HashSet<>();
        for (int i = 2; i <= m; i++) {
            set.add(sum[i-2] % k);
            if (set.contains(sum[i] % k)) return true;
        }
        return false;
    }

    // 525. 连续数组 medium
    public int findMaxLength(int[] nums) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) sum[i] = sum[i - 1] + (nums[i - 1] == 1 ? 1 : -1);
        int ans = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 2; i <= n; i++) {
            if (!map.containsKey(sum[i - 2])) map.put(sum[i-2], i - 2);
            if (map.containsKey(sum[i])) ans = Math.max(ans, i - map.get(sum[i]));
        }
        return ans;
    }

    // 160. 相交链表 easy
    public ListNode getIntersectionNode_daily(ListNode headA, ListNode headB) {
        ListNode dumpA = headA, dumpB = headB;
        while (headA != headB) {
            if (headA == null) headA = dumpB;
            else headA = headA.next;
            if (headB == null) headB = dumpA;
            else headB = headB.next;
        }
        return headA;
    }

    // 203. 移除链表元素 easy
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode temp = dummyHead;
        while (temp.next != null) {
            if (temp.next.val == val) {
                temp.next = temp.next.next;
            } else {
                temp = temp.next;
            }
        }
        return dummyHead.next;
    }

    // 474. 一和零 medium 0-1 bags problem TODO
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][] cnt = new int[len][2];
        for (int i = 0; i < len; i++) {
            String str = strs[i];
            int zero = 0, one = 0;
            for (char c : str.toCharArray()) {
                if (c == '0') zero++;
                else one++;
            }
            cnt[i] = new int[]{zero, one};
        }

        int[][][] f = new int[len + 1][m + 1][n + 1];
        for (int k = 1; k <= len; k++) {
            int zero = cnt[k - 1][0], one = cnt[k - 1][1];
            for (int i = 0; i <= m; i++) {
                for (int j = 0; j <= n; j++) {
                    int a = f[k - 1][i][j];
                    int b = (i >= zero && j >= one) ? f[k - 1][i - zero][j - one] + 1 : 0;
                    f[k][i][j] = Math.max(a, b);
                }
            }
        }
        return f[len][m][n];
    }

    // 494. 目标和 medium
    public int findTargetSumWays_494(int[] nums, int target) {
        return dfs(nums, target, 0, 0);
    }

    private int dfs(int[] nums, int t, int u, int cur) {
        if (u == nums.length) {
            return cur == t ? 1 : 0;
        }
        int left = dfs(nums, t, u + 1, cur + nums[u]);
        int right = dfs(nums, t, u + 1, cur - nums[u]);
        return left + right;
    }

    // 1049. 最后一块石头的重量 II medium
    public int lastStoneWeightII(int[] stones) {
        return 1;
    }
}


public class LeetCode_Hot100 {
    public static void main(String[] args) {
//        int[] arr = {1,2,3};
//        List list = Arrays.asList(1,2,3);
        Solution solution = new Solution();
        int[] nums1 = new int[]{1,2,3};
        int[] nums2 = new int[]{2};
        System.out.println(solution.subsets(nums1));
    }
}
