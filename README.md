# RoboGame

## 相关说明

This is USTC 2024 RoboGame's Isolation Electronic Team's relavent files.

## 负责:
视觉:杨睿卿

## Git教程

相关说明:

1. `<>`表示参数

### 配置

1. **设置用户名**:git config –global user.name `<USERNAME>`
2. **设置用户邮箱**:git config –global user.email `<EMAIL>`
3. **设置编辑器**:git config –global core.editor "code -w"

### 获取远程数据库

1. **克隆远程数据库**：git clone https://github.com/HydrogenOxideLoveHydrogenPeroxide/RoboGame
2. **获取远程数据库提交且合并到当前分支**:git pull
3. **把本地分支推送到远程数据库**:git push

### 本地分支

本地分支分为:**工作目录，索引，对象数据库**三个存储对象。

1. **提交到索引**:git add .
2. **从索引提交到对象数据库，形成提交历史**:git commit -m `<message>`
3. **把提交历史推送到远程数据库**:git push

![1720887891084](image/README/1720887891084.png)
