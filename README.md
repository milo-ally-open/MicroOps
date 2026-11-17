# MicroOps

MicroOps 是一个面向 AI Infra 学习与实践的算子仓库，用于探索 CUDA 与 CANN 并行计算。

## 运行算子

进入对应算子的 `cuda` 或 `cann` 目录后运行：

```bash
python bench.py
```

运行 CANN 算子前，先按开发板环境配置 CANN；具体依赖见对应目录的 README。


## 阶段一：创建练习环境（回滚到起跑线）

- 1. 确保当前环境干净
在开始前，确保当前没有未提交的代码：

```bash 
git status
```

- 2. 查看提交历史，找到起跑线

```bash 
git log --oneline -5 # 找到你想开始练习的那个 commit 哈希值（假设为 abc1234）
```

- 3. 基于历史版本创建并切换到练习分支

```bash 
git checkout -b practice-branch abc1234
```

## 阶段二：自由练习与测试

在这个 practice-branch 分支上，你可以随意修改代码、删除文件、添加新功能。
修改后，正常运行你的测试脚本即可。

## 阶段三：对答案

- 方法1: 不改变你当前的代码，直接让 Git 告诉你，你写的和 main 分支的标准答案有什么不同：

```bash 
git diff main
```

- 方法2: 一键覆盖 (直接看答案)

```bash 
git restore --source=main .
```

- 方法3: 切换查看 

```bash 
git switch main       # 去看答案
git switch practice-branch  # 切回你的练习
```

## 阶段四：清理与收尾

- 场景一: 练完了，不需要保留练习记录
```bash 
# 1. 丢弃练习分支上未提交的修改
git restore .

# 2. 切回主分支
git switch main

# 3. 删除练习分支
git branch -D practice-branch
```

- 场景 2：写出了比标准答案更好的代码！

```bash 
# 1. 提交你的修改
git add .
git commit -m "feat: 我的全新优化实现"

# 2. 切回 main
git switch main

# 3. 合并你的练习分支
git merge practice-branch
```
